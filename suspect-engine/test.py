import logging
import os
from datetime import datetime, timezone
from collections import defaultdict

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from rich.logging import RichHandler


# -----------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# -----------------------------------------------------------
# Environment
# -----------------------------------------------------------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
Database_Name = os.getenv("DATABASE")

edps_claims_collection = os.getenv("EDPS_CLAIMS_COLLECTION")
pharmacy_claims_collection = os.getenv("PHARMACY_CLAIMS_COLLECTION")
eligibility_collection = os.getenv("ELIGIBILITY_COLLECTION")
mbi_crosswalk_collection = os.getenv("MBI_CROSSWALK_COLLECTION")
staging_suspects_collection = os.getenv("STAGING_SUSPECTS_COLLECTION")


# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def safe_date(val):
    if not val:
        return None
    try:
        return val.strftime("%Y-%m-%d")
    except:
        return None


def get_mbi_crosswalk_map(client):
    """Return MemberID → MBI mapping."""
    db = client[Database_Name]
    cursor = db[mbi_crosswalk_collection].find({}, {"_id": 0, "created_dt": 0})

    mapping = {}
    for doc in cursor:
        mid = doc.get("MemberID")
        mbi = doc.get("MBI")
        if mid and mbi:
            mapping[str(mid)] = mbi

    cursor.close()
    return mapping


# -----------------------------------------------------------
# Claim Fetcher (Batch Optimized)
# -----------------------------------------------------------
def batch_fetch_medical_claims(db, mbi_list):
    """
    Fetch all medical claims for a batch of MBIs using $in — huge speed improvement.
    """
    med_map = defaultdict(list)

    cursor = db[edps_claims_collection].find(
        {"Member.Subscriber_ID": {"$in": mbi_list}},
        {
            "_id": 0,
            "Diagnosis.Diag_Codes": 1,
            "ServiceLine": 1,
            "Claim": 1,
            "Type_of_Bill": 1,
            "Provider": 1,
            "Member.Subscriber_ID": 1,
        },
    )

    for doc in cursor:
        key = doc.get("Member", {}).get("Subscriber_ID")
        if key:
            med_map[key].append(doc)

    cursor.close()
    return med_map


def batch_fetch_pharmacy_claims(db, member_ids):
    """
    Fetch pharmacy claims for many members using $in.
    """
    pharm_map = defaultdict(list)

    cursor = db[pharmacy_claims_collection].find(
        {"Member ID": {"$in": member_ids}},
        {
            "_id": 0,
            "Member ID": 1,
            "NDC": 1,
            "Product Label Name": 1,
            "Fill Date": 1,
            "Days Supply": 1,
            "Metric Quantity": 1,
            "Prescriber ID": 1,
            "Prescriber Name": 1,
            "Total Billed": 1,
        },
    )

    for pc in cursor:
        member_id = str(pc.get("Member ID"))
        pharm_map[member_id].append(
            {
                "memberId": member_id,
                "ndc": str(pc.get("NDC")) if pc.get("NDC") else None,
                "drugName": pc.get("Product Label Name"),
                "fillDate": safe_date(pc.get("Fill Date")),
                "daysSupply": pc.get("Days Supply"),
                "quantityDispensed": pc.get("Metric Quantity"),
                "prescriberNPI": pc.get("Prescriber ID"),
                "prescriberName": pc.get("Prescriber Name"),
                "totalBilled": pc.get("Total Billed"),
            }
        )

    cursor.close()
    return pharm_map


# -----------------------------------------------------------
# Bulk Write Helper
# -----------------------------------------------------------
def execute_bulk(collection, ops):
    try:
        result = collection.bulk_write(ops, ordered=False)
        logging.info(
            f"BulkWrite → matched={result.matched_count}, "
            f"modified={result.modified_count}, "
            f"upserted={len(result.upserted_ids)}"
        )
    except BulkWriteError as bwe:
        logging.error(f"BulkWriteError: {bwe.details}")
    except Exception as e:
        logging.error(f"Unexpected bulk write error: {e}")


# -----------------------------------------------------------
# Main Loader
# -----------------------------------------------------------
def load_members_claims_to_database(batch_size=500):
    client = MongoClient(MONGODB_URI)
    db = client[Database_Name]
    suspects_coll = db[staging_suspects_collection]

    logging.info("Loading MBI crosswalk map...")
    mbi_map = get_mbi_crosswalk_map(client)
    logging.info(f"Loaded {len(mbi_map)} MBI entries")

    logging.info("Starting eligibility streaming cursor...")
    cursor = db[eligibility_collection].find({}, {"_id": 0})

    total_processed = 0
    batch = []

    for el in cursor:
        batch.append(el)

        if len(batch) >= batch_size:
            process_batch(batch, db, suspects_coll, mbi_map)
            total_processed += len(batch)
            batch = []

    # final batch
    if batch:
        process_batch(batch, db, suspects_coll, mbi_map)
        total_processed += len(batch)

    cursor.close()
    client.close()

    logging.info(f"Completed processing. Total members processed: {total_processed}")


# -----------------------------------------------------------
# Batch Processor (FAST VERSION)
# -----------------------------------------------------------
def process_batch(batch, db, suspects_coll, mbi_map):

    member_ids = [str(el["memberId"]) for el in batch]
    mbi_list = [mbi_map.get(mid, mid) for mid in member_ids]

    logging.info(f"Fetching claims for batch of {len(batch)} members...")

    # ---- Batch fetch claims (HUGE SPEED-UP)
    medical_map = batch_fetch_medical_claims(db, mbi_list)
    pharmacy_map = batch_fetch_pharmacy_claims(db, member_ids)

    now = datetime.now(timezone.utc)
    ops = []

    for i, el in enumerate(batch):
        mid = member_ids[i]
        mbi_lookup = mbi_list[i]

        doc = {
            "memberId": mid,
            "eligibility": el,  # no undot, much faster
            "medicalClaims": medical_map.get(mbi_lookup, []),
            "pharmacyClaims": pharmacy_map.get(mid, []),
            "updatedAt": now,
            "createdAt": now,
        }

        update = {
            "$set": {
                "eligibility": doc["eligibility"],
                "medicalClaims": doc["medicalClaims"],
                "pharmacyClaims": doc["pharmacyClaims"],
                "updatedAt": now,
            },
            "$setOnInsert": {
                "createdAt": now,
                "memberId": mid,
            },
        }

        ops.append(UpdateOne({"memberId": mid}, update, upsert=True))

        # write in bigger batches for speed
        if len(ops) >= 500:
            execute_bulk(suspects_coll, ops)
            ops = []

    # remaining ops
    if ops:
        execute_bulk(suspects_coll, ops)


# -----------------------------------------------------------
# Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    load_members_claims_to_database(batch_size=500)

