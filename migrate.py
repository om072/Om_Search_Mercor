import turbopuffer
import time
import concurrent.futures
from threading import Lock
import argparse
import requests
import json
import gzip

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOTAL_BATCHES = 39
TURBOPUFFER_REGION = "aws-us-west-2"
TPUF_NAMESPACE_NAME = "om_khangat_tpuf_key"
MAX_RETRIES = 10
NUM_THREADS = 5
STREAMING_ENDPOINT_BASE = "https://mercor-dev--search-eng-interview-documents.modal.run/stream_documents"

TURBOPUFFER_API_KEY = "tpuf_dQHBpZEvl612XAdP0MvrQY5dbS0omPMy"

# Initialize Turbopuffer client
tpuf = turbopuffer.Turbopuffer(
    api_key=TURBOPUFFER_API_KEY,
    region=TURBOPUFFER_REGION,
)
ns = tpuf.namespace(TPUF_NAMESPACE_NAME)

def fetch_and_upsert_batch(batch_number: int):
    endpoint = f"{STREAMING_ENDPOINT_BASE}/{batch_number}"
    logger.info(f"Fetching batch {batch_number} from: {endpoint}")

    try:
        response = requests.get(endpoint, stream=True, timeout=600)
        response.raise_for_status()

        batch = []

        # Collect all chunks of raw gzip data
        compressed_chunks = []
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                compressed_chunks.append(chunk)

        # Combine chunks and decompress
        compressed_data = b"".join(compressed_chunks)

        try:
            # Decompress the gzip data
            decompressed_data = gzip.decompress(compressed_data)

            # Process line by line
            for line in decompressed_data.decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)

                    # Check if it's an error message
                    if "error" in doc:
                        logger.error(f"Error from batch {batch_number}: {doc['error']}")
                        break

                    # Skip documents without embeddings
                    if not doc.get("embedding"):
                        continue

                    profile = {
                        "id": str(doc.get("_id")),
                        "vector": doc.get("embedding", []),
                        "email": doc.get("email", ""),
                        "rerank_summary": doc.get("rerankSummary", ""),
                        "country": doc.get("country", ""),
                        "name": doc.get("name", ""),
                        "linkedin_id": doc.get("linkedinId", ""),
                    }
                    batch.append(profile)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in batch {batch_number}: {e}")
                    continue

        except gzip.BadGzipFile as e:
            logger.error(f"Gzip decompression error in batch {batch_number}: {e}")
            return 0

        if not batch:
            logger.info(f"No documents with embeddings found in batch {batch_number}")
            return 0

        if upsert_batch_to_turbopuffer(batch):
            logger.info(
                f"Successfully processed batch {batch_number}: {len(batch)} documents"
            )
            return len(batch)
        else:
            return 0

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching batch {batch_number}: {e}")
        return 0

def upsert_batch_to_turbopuffer(batch):
    for i in range(MAX_RETRIES):
        try:
            ns.write(
                upsert_rows=batch,
                distance_metric="cosine_distance",
                schema={
                    "id": "string",
                    "rerank_summary": {"type": "string", "full_text_search": True},
                    "email": "string",
                    "country": "string",
                    "name": "string",
                    "linkedin_id": "string",
                },
            )

            logger.info(f"Successfully upserted {len(batch)} documents to Turbopuffer")
            return True
        except Exception as e:
            logger.error(f"Error upserting batch to Turbopuffer: {e}")
            if i < MAX_RETRIES - 1:
                logger.info(f"Retrying in {i + 1} seconds...")
                time.sleep(i + 1)
            else:
                logger.error(f"Turbopuffer upsert failed after {MAX_RETRIES} attempts")
                return False

def delete_namespace():
    try:
        ns.delete_all()
        logger.info("Namespace cleared successfully")
    except Exception as e:
        logger.error(f"Namespace already cleared")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch endpoint to Turbopuffer migration tool"
    )
    parser.add_argument(
        "action",
        choices=["delete", "migrate"],
        help="Action to perform",
        default="migrate",
        nargs="?",
    )
    args = parser.parse_args()

    if args.action == "delete":
        logger.info("Clearing Turbopuffer namespace...")
        delete_namespace()
        exit()

    logger.info("Starting migration from batch endpoints to Turbopuffer")
    logger.info(f"Total batches to process: {TOTAL_BATCHES}")
    logger.info(f"Using {NUM_THREADS} threads")

    batch_numbers = list(range(0, TOTAL_BATCHES))

    total_processed = 0
    lock = Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(fetch_and_upsert_batch, batch_num)
            for batch_num in batch_numbers
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_count = future.result()
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                batch_count = 0

            with lock:
                total_processed += batch_count
            logger.info(f"Total processed so far: {total_processed}")

    logger.info(f"Migration completed! Total documents processed: {total_processed}")