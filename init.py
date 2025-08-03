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
# TP_API_KEY = os.getenv("TURBOPUFFER_API_KEY") # No longer needed for Turbopuffer

# --- NEW: Local FAISS Index Paths ---
FAISS_INDEX_PATH = "linkedin_profiles.faiss"
ID_MAPPING_PATH = "linkedin_ids_to_index_mapping.pkl"
INDEX_TO_ID_PATH = "linkedin_index_to_ids.pkl"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_mongo_collection_and_count(uri: str, db_name: str, collection_name: str):
    """Connects to MongoDB and returns the collection and total document count."""
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        total_documents = collection.count_documents({})
        logging.info(f"Connected to MongoDB. Total documents to index: {total_documents}")
        return collection, total_documents
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None, 0


# Removed _upsert_batch_task as it's specific to Turbopuffer

def build_faiss_index_batch(mongo_collection, batch_size=10000):
    """
    Builds a FAISS index from document embeddings using batch processing.
    Much faster and memory efficient for large datasets.
    """
    # Initialize storage
    all_embeddings = []
    all_mongo_ids = []
    all_metadata = []
    
    # Get total count
    total_docs = mongo_collection.count_documents({})
    logging.info(f"Starting batch processing of {total_docs} documents with batch size {batch_size}")
    
    processed_count = 0
    batch_count = 0
    
    # Process in batches
    cursor = mongo_collection.find({}).batch_size(batch_size)
    
    for doc in cursor:
        if "embedding" in doc and doc["embedding"] and "_id" in doc:
            all_embeddings.append(doc["embedding"])
            all_mongo_ids.append(str(doc["_id"]))
            
            # Extract metadata
            experience_titles = [p.get("title", "") for exp in doc.get("experience", []) for p in
                                 exp.get("positions", []) if p.get("title")]
            education_degrees = [d.get("degree", "") for d in doc.get("education", {}).get("degrees", []) if d.get("degree")]
            education_fields = [d.get("fieldOfStudy", "") for d in doc.get("education", {}).get("degrees", []) if d.get("fieldOfStudy")]
            
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
                "education_fields": ",".join(education_fields),
                "skills": ",".join([skill for skill in doc.get("skills", []) if skill]),
            })
            
        processed_count += 1
        
        # Log progress every batch
        if processed_count % batch_size == 0:
            batch_count += 1
            logging.info(f"Processed batch {batch_count}: {processed_count}/{total_docs} documents ({processed_count/total_docs*100:.1f}%)")
    
    logging.info(f"Finished processing all documents. Valid embeddings: {len(all_embeddings)}")
    
    if not all_embeddings:
        logging.error("No valid embeddings found to build FAISS index.")
        return
    
    # Build FAISS index
    embeddings_np = np.array(all_embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    
    logging.info(f"Building FAISS index with {len(embeddings_np)} vectors of dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    logging.info("FAISS index built successfully.")
    
    # Save index and mappings
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Create mappings
    index_to_mongo_id = {i: all_mongo_ids[i] for i in range(len(all_mongo_ids))}
    mongo_id_to_metadata = {all_mongo_ids[i]: all_metadata[i] for i in range(len(all_mongo_ids))}
    
    with open(ID_MAPPING_PATH, 'wb') as f:
        pickle.dump(index_to_mongo_id, f)
    with open(INDEX_TO_ID_PATH, 'wb') as f:
        pickle.dump(mongo_id_to_metadata, f)
    
    logging.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
    logging.info(f"ID mappings and metadata saved to {ID_MAPPING_PATH} and {INDEX_TO_ID_PATH}")
    logging.info(f"Successfully indexed {len(embeddings_np)} documents with FAISS")


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
                                 exp.get("positions", []) if p.get("title")]
            education_degrees = [d.get("degree", "") for d in doc.get("education", {}).get("degrees", []) if d.get("degree")]
            education_fields = [d.get("fieldOfStudy", "") for d in doc.get("education", {}).get("degrees", []) if d.get("fieldOfStudy")]

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
                "skills": ",".join([skill for skill in doc.get("skills", []) if skill]),  # Add skills if needed
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
    logging.info(f"ID mappings and metadata saved to {ID_MAPPING_PATH}")


# Removed verify_counts as it's Turbopuffer specific, FAISS is local file check.

if __name__ == "__main__":
    logging.info("Starting the data ingestion and FAISS index building pipeline...")

    mongo_collection, total_count = get_mongo_collection_and_count(MONGO_URI, DB_NAME, COLLECTION_NAME)

    if mongo_collection is not None and total_count > 0:
        # Use batch processing for much faster and memory-efficient ingestion
        build_faiss_index_batch(mongo_collection, batch_size=10000)
        # Verification for FAISS is checking file existence and size.
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAPPING_PATH):
            logging.info("FAISS index and ID mappings/metadata files exist locally.")
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