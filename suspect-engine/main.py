import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from rich.logging import RichHandler

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
Database_Name = os.getenv("DATABASE")
edps_claims_collection = os.getenv("EDPS_CLAIMS_COLLECTION")
pharmacy_claims_collection = os.getenv("PHARMACY_CLAIMS_COLLECTION")
eligibility_collection = os.getenv("ELIGIBILITY_COLLECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o-mini"
llm_client = OpenAI(api_key=OPENAI_API_KEY)



def load_members_with_claims_from_docs(eligibility_docs):
    """Given a list of eligibility documents, fetch medical & pharmacy claims."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[Database_Name]
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return []

    members = []

    for el in eligibility_docs:
        member_id = el["memberId"]

        # EDPS medical claims
        medical_claims_cursor = db[edps_claims_collection].find(
            {"Member.Subscriber_ID": member_id},
            {
                "_id": 0,
                "Diagnosis.Diag_Codes": 1,
                "ServiceLine.LXServiceNo": 1,
                "ServiceLine.BilledCPT_Code": 1,
                "ServiceLine.Billed_CPTDesc": 1,
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
        medical_claims = list(medical_claims_cursor)

        # Pharmacy claims
        pharmacy_claims_cursor = db[pharmacy_claims_collection].find(
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

        pharmacy_claims = [
            {
                "memberId": pc.get("Member ID"),
                "ndc": str(pc.get("NDC")),
                "drugName": pc.get("Product Label Name"),
                "fillDate": (
                    pc.get("Fill Date").strftime("%Y-%m-%d")
                    if pc.get("Fill Date")
                    else None
                ),
                "daysSupply": pc.get("Days Supply"),
                "quantityDispensed": pc.get("Metric Quantity"),
                "prescriberNPI": pc.get("Prescriber ID"),
                "prescriberName": pc.get("Prescriber Name"),
                "totalBilled": pc.get("Total Billed"),
            }
            for pc in pharmacy_claims_cursor
        ]

        members.append(
            {
                "memberId": member_id,
                "eligibility": el,
                "medicalClaims": medical_claims,
                "pharmacyClaims": pharmacy_claims,
            }
        )

    return members


def call_llm_for_suspects(members):
    """Calls the LLM to generate suspects for a batch of members."""
    prompt = f"""
You are a clinical AI assistant. Identify possible 'suspects' (undiagnosed or missing chronic conditions) 
for the following members using their medical claims, pharmacy claims, and eligibility data.

Return results in JSON exactly like this:
[
  {{
    "memberId": "...",
    "suspectType": "...",
    "suspectDiagnosis": {{
      "code": "...",
      "description": "...",
      "hccCategory": "..."
    }},
    "confidenceScore": 0.85,
    "priority": "...",
    "evidence": {{
      "summary": "...",
      "details": ["...", "..."]
    }},
    "suggestedAction": "..."
  }}
]

Members: {members}
"""
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        llm_output_text = response.choices[0].message.content

        suspects = json.loads(llm_output_text)
        return suspects

    except json.JSONDecodeError:
        logging.error("Failed to decode LLM response")
        logging.error(llm_output_text)
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []


def save_suspects_to_mongo(suspects):
    """Saves suspect records to MongoDB collection ui.member.suspects"""
    if not suspects:
        logging.info("No suspects to save")
        return

    try:
        client = MongoClient(MONGODB_URI)
        db = client[Database_Name]
        db["ui.member.suspects"].insert_many(suspects)
        logging.info(f"Saved {len(suspects)} suspect records to ui.member.suspects")
    except Exception as e:
        logging.error(f"Error saving suspects: {e}")


def process_all_members(batch_size: int = 100):
    """Process the entire eligibility collection in batches, call LLM, and save suspects."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[Database_Name]
        logging.info("Connected to MongoDB")
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return

    cursor = db[eligibility_collection].find({}, {"_id": 0, "createdAt": 0, "updatedAt": 0})
    batch = []
    total_processed = 0

    for el in cursor:
        batch.append(el)
        if len(batch) >= batch_size:
            logging.info(f"Processing batch of {len(batch)} members...")
            members_with_claims = load_members_with_claims_from_docs(batch)
            suspects = call_llm_for_suspects(members_with_claims)
            save_suspects_to_mongo(suspects)
            total_processed += len(batch)
            logging.info(f"Total members processed so far: {total_processed}")
            batch = []

    # Process any remaining members
    if batch:
        logging.info(f"Processing final batch of {len(batch)} members...")
        members_with_claims = load_members_with_claims_from_docs(batch)
        suspects = call_llm_for_suspects(members_with_claims)
        save_suspects_to_mongo(suspects)
        total_processed += len(batch)
        logging.info(f"Total members processed: {total_processed}")


if __name__ == "__main__":
    logging.info("Starting batch processing for all members...")
    process_all_members(batch_size=100)
    logging.info("Processing complete.")

