import os
import logging
# --- NEW: Import faiss and pickle ---
import faiss
import pickle
from dotenv import load_dotenv
import numpy as np
from typing import Dict, Any, List

# Import our custom utilities
from utils.openai_utils import QueryRewriter
from utils.voyageai_utils import VoyageAIClient
from utils.string_utils import parseCommaSeparatedIDs # Needed for filtering strings

# Load environment variables
load_dotenv()
# TP_API_KEY = os.getenv("TP_API_KEY") # No longer needed for Turbopuffer

# --- NEW: Local FAISS Index Paths ---
FAISS_INDEX_PATH = "linkedin_profiles.faiss"
ID_MAPPING_PATH = "linkedin_ids_to_index_mapping.pkl"  # This now holds index -> mongo_id mapping
METADATA_MAPPING_PATH = "linkedin_index_to_ids.pkl"  # This now holds mongo_id -> metadata_dict mapping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SearchEngine:
    def __init__(self):
        # --- NEW: Load local FAISS index and ID mappings ---
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(ID_MAPPING_PATH, 'rb') as f:
                self.index_to_mongo_id = pickle.load(f)
            with open(METADATA_MAPPING_PATH, 'rb') as f:
                self.mongo_id_to_metadata = pickle.load(f)
            logging.info("Successfully loaded local FAISS index and ID mappings/metadata.")
        except Exception as e:
            logging.error(f"Failed to load local FAISS index or mappings: {e}. Run init.py first.")
            # Critical error, prevent further execution without index
            raise RuntimeError("FAISS index setup failed.")

        self.query_rewriter = QueryRewriter()
        self.voyage_client = VoyageAIClient()

    def _get_query_embedding(self, query: str) -> List[float]:
        # ... (remains the same) ...
        try:
            return self.voyage_client.embed_query(query)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get query embedding: {e}")

    def search(self, query: str, hard_criteria: dict, soft_criteria: dict) -> List[str]:
        """
        Performs a hybrid search: Vector Search -> Post-Filtering (Hard Criteria) -> Re-ranking (Soft Criteria).
        """
        logging.info(f"Processing query: '{query}'")

        # Step 1: Use OpenAI to understand the query and extract attributes
        rewritten_query = self.query_rewriter.rewrite_query(query)
        logging.info(f"OpenAI-rewritten query attributes: {rewritten_query}")

        final_hard_filters = {**hard_criteria, **rewritten_query}

        # Step 2: Vector Search (Semantic Similarity) - FAISS retrieves all candidates first
        try:
            query_vector = self._get_query_embedding(query)
            query_vector_np = np.array([query_vector]).astype('float32')  # FAISS expects a 2D array
        except RuntimeError as e:
            logging.error(f"Search failed due to embedding error: {e}")
            return []

        # Query FAISS for the top N most similar vectors
        # Retrieve more candidates than 'top_k' if you expect many to be filtered out.
        # A common practice is to retrieve top 1000 or 10000 for filtering.
        D, I = self.index.search(query_vector_np, 1000)  # D are distances, I are internal FAISS indices

        logging.info(f"Retrieved {len(I[0])} candidates from FAISS vector search.")

        # Step 3: Post-Filtering (Hard Criteria)
        # Iterate through the retrieved candidates and apply hard filters using their metadata
        filtered_candidates_with_meta = []
        for faiss_idx in I[0]:
            mongo_id = self.index_to_mongo_id.get(faiss_idx)
            if mongo_id is None:
                logging.warning(f"FAISS index {faiss_idx} has no corresponding MongoDB ID. Skipping.")
                continue

            metadata = self.mongo_id_to_metadata.get(mongo_id)
            if metadata is None:
                logging.warning(f"No metadata found for MongoDB ID: {mongo_id}. Skipping for filtering.")
                continue

            # Apply hard filters dynamically
            passes_hard_filters = True
            if final_hard_filters.get("yearsOfWorkExperience"):
                if metadata.get("yearsOfWorkExperience", 0.0) < final_hard_filters["yearsOfWorkExperience"]:
                    passes_hard_filters = False

            if passes_hard_filters and final_hard_filters.get("experience_titles"):
                # Use string_utils.ParseCommaSeparatedIDs for robust checking
                candidate_titles = parseCommaSeparatedIDs(metadata.get("experience_titles", ""))
                # Check if ANY of the candidate's titles contains the filter keyword
                filter_keyword = final_hard_filters["experience_titles"].lower()
                if not any(filter_keyword in title.lower() for title in candidate_titles):
                    passes_hard_filters = False

            if passes_hard_filters and final_hard_filters.get("education_degrees"):
                candidate_degrees = parseCommaSeparatedIDs(metadata.get("education_degrees", ""))
                filter_keyword = final_hard_filters["education_degrees"].lower()
                if not any(filter_keyword in degree.lower() for degree in candidate_degrees):
                    passes_hard_filters = False

            # Add more hard filters here based on your queries and metadata fields
            # e.g., for 'location', 'education_fields', 'skills', etc.

            if passes_hard_filters:
                # Add original FAISS score for re-ranking
                filtered_candidates_with_meta.append({
                    'id': mongo_id,
                    'vector_score': D[0][np.where(I[0] == faiss_idx)[0][0]],  # Get original similarity score
                    'metadata': metadata
                })

        logging.info(f"Filtered down to {len(filtered_candidates_with_meta)} candidates after hard filtering.")

        # Step 4: Fallback strategy if too few candidates
        if len(filtered_candidates_with_meta) < 10:
            logging.warning(f"Only {len(filtered_candidates_with_meta)} candidates found with strict filtering. Applying relaxed filtering.")
            # Try relaxed filtering to get more legitimate candidates
            additional_candidates = self._apply_relaxed_filtering(I[0], D[0], final_hard_filters, filtered_candidates_with_meta)
            filtered_candidates_with_meta.extend(additional_candidates)
            logging.info(f"After relaxed filtering: {len(filtered_candidates_with_meta)} total candidates.")
        
        # Step 5: Re-ranking (Soft Criteria)
        if not filtered_candidates_with_meta:
            # Last resort: return top vector similarity candidates
            logging.warning("No candidates found even with relaxed filtering. Using top vector similarity results.")
            return self._get_top_vector_candidates(I[0], D[0], 10)

        reranked_candidates = []
        for candidate_data in filtered_candidates_with_meta:
            meta = candidate_data['metadata']
            vector_score = candidate_data['vector_score']

            prestige_weight = soft_criteria.get('prestige_score_weight', 0.5)
            experience_weight = soft_criteria.get('experience_experience_weight',
                                                  0.5)  # Typo fixed from query spreadsheet

            prestige = meta.get('prestigeScore', 0)
            experience = meta.get('yearsOfWorkExperience', 0)

            # --- Re-ranking Logic ---
            # You might need to normalize prestige/experience if they have very different scales
            # than the vector score, especially if you tune weights.
            # For FAISS L2 distance, smaller scores are better (closer). We need to invert or scale.
            # A common way is to make similarity scores positive and larger for better matches.
            # Or use 1 - (distance / max_distance)

            # Simple inverse of distance for L2: larger score for smaller distance
            similarity_score = 1.0 / (1.0 + vector_score)  # Ensure no division by zero or large negative score

            final_score = (similarity_score * 1.0) + \
                          (prestige * prestige_weight) + \
                          (experience * experience_weight)

            reranked_candidates.append({
                'id': candidate_data['id'],
                'final_score': final_score
            })

        reranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return [candidate['id'] for candidate in reranked_candidates[:10]]

    def _apply_relaxed_filtering(self, faiss_indices, distances, hard_filters, existing_candidates):
        """Apply more lenient filtering to find additional legitimate candidates."""
        existing_ids = {c['id'] for c in existing_candidates}
        relaxed_candidates = []
        
        # Relax filtering criteria - use partial matching instead of exact matching
        for i, faiss_idx in enumerate(faiss_indices):
            if len(relaxed_candidates) + len(existing_candidates) >= 10:
                break
                
            mongo_id = self.index_to_mongo_id.get(faiss_idx)
            if not mongo_id or mongo_id in existing_ids:
                continue
                
            metadata = self.mongo_id_to_metadata.get(mongo_id)
            if not metadata:
                continue
            
            # Apply relaxed criteria
            passes_relaxed_filters = True
            
            # For experience titles, try broader matching
            if hard_filters.get("experience_titles"):
                candidate_titles = parseCommaSeparatedIDs(metadata.get("experience_titles", ""))
                filter_keywords = hard_filters["experience_titles"].lower().split()
                # Check if ANY keyword matches ANY title (more lenient)
                if candidate_titles and not any(
                    any(keyword in title.lower() for keyword in filter_keywords) 
                    for title in candidate_titles
                ):
                    passes_relaxed_filters = False
            
            # For education, be more lenient about partial matches
            if passes_relaxed_filters and hard_filters.get("education_degrees"):
                candidate_degrees = parseCommaSeparatedIDs(metadata.get("education_degrees", ""))
                education_fields = parseCommaSeparatedIDs(metadata.get("education_fields", ""))
                filter_keywords = hard_filters["education_degrees"].lower().split()
                
                # Check both degrees and fields of study
                has_education_match = False
                if candidate_degrees:
                    has_education_match = any(
                        any(keyword in degree.lower() for keyword in filter_keywords)
                        for degree in candidate_degrees
                    )
                if not has_education_match and education_fields:
                    has_education_match = any(
                        any(keyword in field.lower() for keyword in filter_keywords)
                        for field in education_fields
                    )
                if not has_education_match:
                    passes_relaxed_filters = False
            
            # Reduce experience requirements by 50%
            if passes_relaxed_filters and hard_filters.get("yearsOfWorkExperience"):
                min_exp = hard_filters["yearsOfWorkExperience"] * 0.5  # Reduce requirement
                if metadata.get("yearsOfWorkExperience", 0.0) < min_exp:
                    passes_relaxed_filters = False
            
            if passes_relaxed_filters:
                relaxed_candidates.append({
                    'id': mongo_id,
                    'vector_score': distances[i],
                    'metadata': metadata
                })
        
        logging.info(f"Relaxed filtering found {len(relaxed_candidates)} additional candidates.")
        return relaxed_candidates
    
    def _get_top_vector_candidates(self, faiss_indices, distances, count):
        """Get top candidates based purely on vector similarity as fallback."""
        vector_candidates = []
        
        for i, faiss_idx in enumerate(faiss_indices[:count * 3]):  # Check more to ensure we get valid ones
            if len(vector_candidates) >= count:
                break
                
            mongo_id = self.index_to_mongo_id.get(faiss_idx)
            if mongo_id and mongo_id in self.mongo_id_to_metadata:
                vector_candidates.append(mongo_id)
        
        logging.info(f"Vector fallback returned {len(vector_candidates)} candidates.")
        return vector_candidates[:count]