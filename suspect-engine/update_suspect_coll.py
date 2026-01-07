import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from bson.objectid import ObjectId
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
Database_Name = os.getenv("DATABASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o-mini"
llm_client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm_for_suspects(members):
    """Calls the LLM to generate suspects for a batch of members."""
    prompt2 = f"""
      You are a clinical AI assistant.

  Return your answer as **strict raw JSON only**.
  Do NOT include markdown formatting, backticks, comments, or explanations.
  The response MUST be valid JSON and MUST NOT contain ```json or ```.

  Only return diagnosis codes (ICD-10) that map to **V28 Medicare Advantage Risk Adjustment Model HCCs**.  
  Do not include any diagnosis codes that are not part of the V28 HCC model.

  Output format:
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
                {"role": "user", "content": prompt2},
            ],
            temperature=1.0,
        )

        output_text = response.choices[0].message.content
        return json.loads(output_text)

    except json.JSONDecodeError:
        logging.error("Failed to decode LLM response")
        logging.error(output_text)
        return []

    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []


def save_suspects_to_mongo(suspects):
    if not suspects:
        return

    try:
        client = MongoClient(MONGODB_URI)
        db = client[Database_Name]
        col = db["ui.member.suspects"]

        now = datetime.now(timezone.utc)
        operations = []

        for s in suspects:
            member_id = s.get("memberId")
            if not member_id:
                logging.warning(f"Skipping suspect without memberId: {s}")
                continue

            operations.append(
                UpdateOne(
                    {"memberId": member_id},
                    {
                        "$set": {**s, "updatedAt": now},
                        "$setOnInsert": {"createdAt": now},
                    },
                    upsert=True,
                )
            )

        if operations:
            result = col.bulk_write(operations, ordered=False)
            logging.info(
                f"Written: matched={result.matched_count}, modified={result.modified_count}, upserted={len(result.upserted_ids)}"
            )

    except Exception as e:
        logging.error(f"Error saving suspects: {e}")


def load_members_from_staging(skip, limit):
    """Pulls documents from ui.stg.suspects with paging."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[Database_Name]

        staging = db["ui.stg.suspects"]

        docs = list(
            staging.find(
                {"pharmacyClaims": {"$ne": []}, "medicalClaims": {"$ne": []}},
                {"_id": 0, "updatedAt": 0, "createdAt": 0},
            )
            .skip(skip)
            .limit(limit)
        )

        return docs

    except Exception as e:
        logging.error(f"Error loading staging members: {e}")
        return []


def process_batch(members):
    suspects = call_llm_for_suspects(members)
    save_suspects_to_mongo(suspects)
    return len(members)


def process_all_members(batch_size=100, max_workers=8):
    client = MongoClient(MONGODB_URI)
    db = client[Database_Name]
    staging = db["ui.stg.suspects"]

    total_docs = staging.count_documents(
        {"pharmacyClaims": {"$ne": []}, "medicalClaims": {"$ne": []}}
    )

    logging.info(f"Total staging records to process: {total_docs}")

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    skip = 0
    total_processed = 0

    while skip < total_docs:
        members = load_members_from_staging(skip, batch_size)
        if not members:
            break

        futures.append(executor.submit(process_batch, members))

        skip += batch_size
        logging.info(f"Submitted batch, skip={skip}")

        if len(futures) > max_workers * 3:
            for f in as_completed(futures):
                total_processed += f.result()
            futures = []

    for f in as_completed(futures):
        total_processed += f.result()

    executor.shutdown(wait=True)
    logging.info(f"Processing complete. Total processed: {total_processed}")


if __name__ == "__main__":
    logging.info("Starting batch processing...")
    process_all_members(batch_size=4)
    logging.info("All Done!")
