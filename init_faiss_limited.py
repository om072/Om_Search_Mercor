import os
import logging
from pymongo import MongoClient
# --- NEW: Import faiss and pickle ---
import faiss
import pickle
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "interview_data"
COLLECTION_NAME = "linkedin_data_subset"

# --- NEW: Local FAISS Index Paths ---
FAISS_INDEX_PATH = "linkedin_profiles.faiss"
ID_MAPPING_PATH = "linkedin_ids_to_index_mapping.pkl"
INDEX_TO_ID_PATH = "linkedin_index_to_ids.pkl"

# Limit the number of documents to process for faster testing
MAX_DOCUMENTS = 5000

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_mongo_collection_and_count(uri: str, db_name: str, collection_name: str):
    """Connects to MongoDB and returns the collection and total document count."""
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        total_documents = collection.count_documents({})
        logging.info(f"Connected to MongoDB. Total documents available: {total_documents}")
        return collection, total_documents
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None, 0


def build_faiss_index(documents: list):
    """
    Builds a FAISS index from document embeddings and saves it locally.
    Also saves mappings between internal FAISS indices and MongoDB _ids.
    """
    if not documents:
        logging.warning("No documents to index. FAISS index building skipped.")
        return

    embeddings = []
    mongo_ids = []  # To store the original MongoDB _id for each vector

    # Store all relevant metadata locally as well, as FAISS only stores vectors.
    # We'll need this metadata for post-filtering and re-ranking.
    all_metadata = []

    logging.info(f"Extracting embeddings and metadata from {len(documents)} documents...")
    for i, doc in enumerate(documents):
        if "embedding" in doc and doc["embedding"] and "_id" in doc:
            embeddings.append(doc["embedding"])
            mongo_ids.append(str(doc["_id"]))  # Store MongoDB ObjectId as string

            # Store metadata locally that FAISS won't handle directly
            experience_titles = [p.get("title", "") for exp in doc.get("experience", []) for p in
                                 exp.get("positions", [])]
            education_degrees = [d.get("degree", "") for d in doc.get("education", {}).get("degrees", [])]
            education_fields = [d.get("fieldOfStudy", "") for d in doc.get("education", {}).get("degrees", [])]

            all_metadata.append({
                "personId": doc.get("personId", ""),
                "name": doc.get("name", ""),
                "yearsOfWorkExperience": doc.get("yearsOfWorkExperience", 0.0),
                "prestigeScore": doc.get("prestigeScore", 0.0),
                "rerankSummary": doc.get("rerankSummary", ""),
                "headline": doc.get("headline", ""),
                "country": doc.get("country", ""),
                "experience_titles": ",".join(experience_titles),
                "education_degrees": ",".join(education_degrees),
                "education_fields": ",".join(education_fields),  # Add this field as well
                "skills": ",".join(doc.get("skills", [])),  # Add skills if needed
            })
        else:
            logging.warning(f"Skipping document with invalid data (missing embedding or _id): {doc.get('_id', 'N/A')}")

    if not embeddings:
        logging.error("No valid embeddings found to build FAISS index.")
        return

    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]

    logging.info(f"Building FAISS index with {len(embeddings_np)} vectors of dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean)
    index.add(embeddings_np)
    logging.info("FAISS index built.")

    # Save the index and the mappings locally
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Create a mapping from FAISS internal index ID to MongoDB _id
    index_to_mongo_id = {i: mongo_ids[i] for i in range(len(mongo_ids))}

    # Save all metadata indexed by MongoDB _id (for easy lookup later)
    # Create a dictionary {mongo_id: metadata_dict}
    mongo_id_to_metadata = {mongo_ids[i]: all_metadata[i] for i in range(len(mongo_ids))}

    with open(ID_MAPPING_PATH, 'wb') as f:
        pickle.dump(index_to_mongo_id, f)
    with open(INDEX_TO_ID_PATH, 'wb') as f:  # Corrected from previous thought
        pickle.dump(mongo_id_to_metadata, f)

    logging.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
    logging.info(f"ID mappings and metadata saved to {ID_MAPPING_PATH} and {INDEX_TO_ID_PATH}")
    logging.info(f"Successfully indexed {len(embeddings_np)} documents with FAISS")


if __name__ == "__main__":
    logging.info("Starting the limited data ingestion and FAISS index building pipeline...")
    logging.info(f"Will process maximum {MAX_DOCUMENTS} documents")

    mongo_collection, total_count = get_mongo_collection_and_count(MONGO_URI, DB_NAME, COLLECTION_NAME)

    if mongo_collection is not None and total_count > 0:
        # Limit the number of documents to fetch
        docs_to_fetch = min(MAX_DOCUMENTS, total_count)
        logging.info(f"Attempting to fetch {docs_to_fetch} documents from MongoDB for FAISS indexing...")
        
        # Use limit() to fetch only the specified number of documents
        documents = list(mongo_collection.find({}).limit(docs_to_fetch))
        logging.info(f"Successfully fetched {len(documents)} documents into memory for FAISS.")

        build_faiss_index(documents)
        # Verification for FAISS is checking file existence and size.
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAPPING_PATH):
            logging.info("FAISS index and ID mappings/metadata files exist locally.")
            logging.info("Pipeline setup complete successfully!")
        else:
            logging.error("FAILURE: FAISS index or ID mapping/metadata files were not created.")
    else:
        if mongo_collection is None:
            logging.error(
                "Pipeline failed to start: MongoDB collection could not be obtained (check connection/credentials).")
        elif total_count == 0:
            logging.warning("Pipeline started but found 0 documents in MongoDB collection. Nothing to index.")
        else:
            logging.error("Pipeline failed to start due to an unknown issue.")

    logging.info("Pipeline setup complete.")