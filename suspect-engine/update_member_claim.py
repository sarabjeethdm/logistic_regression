import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

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
# Utility functions
# -----------------------------------------------------------


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


def undot_keys(document):
    """Convert dotted keys to nested dictionaries (recursive)."""
    if isinstance(document, list):
        return [undot_keys(item) for item in document]

    if not isinstance(document, dict):
        return document

    new_doc = {}
    for key, value in document.items():
        if "." in key:
            parts = key.split(".")
            current = new_doc
            for p in parts[:-1]:
                current = current.setdefault(p, {})
            current[parts[-1]] = undot_keys(value)
        else:
            new_doc[key] = undot_keys(value)

    return new_doc


def safe_date(val):
    if not val:
        return None
    try:
        return val.strftime("%Y-%m-%d")
    except:
        return None


# -----------------------------------------------------------
# Claim Fetcher
# -----------------------------------------------------------


def fetch_member_claims(el, db, mbi_map):
    """Fetch medical & pharmacy claims for a single member."""

    member_id = str(el.get("memberId"))
    mbi_lookup = mbi_map.get(member_id, member_id)

    # --- Medical Claims ---
    med_cursor = db[edps_claims_collection].find(
        {"Member.Subscriber_ID": mbi_lookup},
        {
            "_id": 0,
            "Diagnosis.Diag_Codes": 1,
            "ServiceLine.LXServiceNo": 1,
            "ServiceLine.BilledCPT_Code": 1,
            "ServiceLine.BilledCPTDesc": 1,
            "ServiceLine.Line_SvcDate": 1,
            "Claim.ClaimID": 1,
            "Claim.POS": 1,
            "Type_of_Bill": 1,
            "Provider.BillProv_NPI": 1,
            "Provider.BillProv_LastName": 1,
            "Member.Subscriber_ID": 1,
            "Member.Subscriber_DOB": 1,
            "Member.Subscriber_Gender": 1,
        },
    )

    medical_claims = [undot_keys(c) for c in med_cursor]
    med_cursor.close()

    # --- Pharmacy Claims ---
    pharm_cursor = db[pharmacy_claims_collection].find(
        {"Member ID": member_id},
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

    pharmacy_claims = []
    for pc in pharm_cursor:
        pharmacy_claims.append(
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
    pharm_cursor.close()

    now = datetime.now(timezone.utc)

    return {
        "memberId": member_id,
        "eligibility": undot_keys(el),
        "medicalClaims": medical_claims,
        "pharmacyClaims": pharmacy_claims,
        "updatedAt": now,
        "createdAt": now,
    }


# -----------------------------------------------------------
# Batch Processor
# -----------------------------------------------------------


def process_batch(batch, db, suspects_coll, mbi_map, max_workers):
    ops = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_member_claims, el, db, mbi_map) for el in batch
        ]

        for future in as_completed(futures):
            try:
                doc = future.result()

                query = {"memberId": doc["memberId"]}
                update = {
                    "$set": {
                        "eligibility": doc["eligibility"],
                        "medicalClaims": doc["medicalClaims"],
                        "pharmacyClaims": doc["pharmacyClaims"],
                        "updatedAt": doc["updatedAt"],
                    },
                    "$setOnInsert": {
                        "createdAt": doc["createdAt"],
                        "memberId": doc["memberId"],
                    },
                }

                ops.append(UpdateOne(query, update, upsert=True))

                if len(ops) >= 50:
                    execute_bulk(suspects_coll, ops)
                    ops = []

            except Exception as e:
                logging.error(f"Thread error while fetching claims: {e}")

    if ops:
        execute_bulk(suspects_coll, ops)


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


def load_members_claims_to_database(batch_size=100, max_workers=8):
    client = MongoClient(MONGODB_URI)
    db = client[Database_Name]
    suspects_coll = db[staging_suspects_collection]

    logging.info("Loading MBI crosswalk map...")
    mbi_map = get_mbi_crosswalk_map(client)
    logging.info(f"Loaded {len(mbi_map)} MBI entries")

    # -----------------------------------------------
    # FIX: CursorNotFound eliminated
    # We fully read eligibility records into memory
    # -----------------------------------------------
    logging.info("Fetching all eligibility records into memory...")
    eligibility_docs = list(db[eligibility_collection].find({}, {"_id": 0}))
    logging.info(f"Total eligibility records loaded: {len(eligibility_docs)}")

    total_processed = 0
    batch = []

    for el in eligibility_docs:
        batch.append(el)

        if len(batch) >= batch_size:
            logging.info(f"Processing batch of {len(batch)}...")
            process_batch(batch, db, suspects_coll, mbi_map, max_workers)
            total_processed += len(batch)
            batch = []

    # Final batch
    if batch:
        logging.info(f"Processing final batch of {len(batch)}...")
        process_batch(batch, db, suspects_coll, mbi_map, max_workers)
        total_processed += len(batch)

    logging.info(f"Completed processing. Total members processed: {total_processed}")
    client.close()


# -----------------------------------------------------------
# Entry Point
# -----------------------------------------------------------

if __name__ == "__main__":
    load_members_claims_to_database(batch_size=100, max_workers=8)
