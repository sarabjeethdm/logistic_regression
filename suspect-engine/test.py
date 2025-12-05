import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from rich.logging import RichHandler

logging.basicConfig(
    # level="NOTSET",
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

load_dotenv()

# MongoDB config
MONGODB_URI = os.getenv("MONGODB_URI")
Database_Name = os.getenv("DATABASE")
edps_claims_collection = os.getenv("EDPS_CLAIMS_COLLECTION")
pharmacy_claims_collection = os.getenv("PHARMACY_CLAIMS_COLLECTION")
eligibility_collection = os.getenv("ELIGIBILITY_COLLECTION")
member_suspect_collection = os.getenv("UI_MEMBER_SUSPECTS_COLLECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI config
LLM_MODEL = "gpt-4o-mini"
llm_client = OpenAI(api_key=OPENAI_API_KEY)


def load_members_with_claims(batch_size: int = 100):
    """Pulls member + medical claims + pharmacy claims from the 3 collections."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[Database_Name]
        logging.info("Connected to MongoDB")
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return []

    eligibility_docs = list(
        db[eligibility_collection]
        .find({}, {"_id": 0, "createdAt": 0, "updatedAt": 0})
        .limit(batch_size)
    )
    logging.info(f"Pulled {len(eligibility_docs)} members from eligibility collection")

    members = []

    for el in eligibility_docs:
        member_id = el["memberId"]
        logging.info(f"Processing member ID: {member_id}")

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
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a clinical assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    llm_output_text = response.choices[0].message.content

    try:
        suspects = json.loads(llm_output_text)
    except json.JSONDecodeError:
        logging.error("Failed to decode LLM response")
        logging.error(llm_output_text)
        suspects = []

    return suspects


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


if __name__ == "__main__":
    logging.info("Starting to load members with claims...")
    member_data = load_members_with_claims(batch_size=10)
    logging.info(f"Loaded data for {len(member_data)} members")

    logging.info("Calling LLM to generate suspects...")
    suspects = call_llm_for_suspects(member_data)

    logging.info(f"LLM returned {len(suspects)} suspect records")
    save_suspects_to_mongo(suspects)
