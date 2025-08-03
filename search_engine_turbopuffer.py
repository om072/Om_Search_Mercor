import os
import logging
import turbopuffer
from dotenv import load_dotenv
import numpy as np
from typing import Dict, Any, List

# Import our custom utilities
from utils.openai_utils import QueryRewriter
from utils.voyageai_utils import VoyageAIClient
from utils.string_utils import parseCommaSeparatedIDs # Needed for filtering strings

# Load environment variables
load_dotenv()

# Turbopuffer configuration
TURBOPUFFER_REGION = "aws-us-west-2"
TPUF_NAMESPACE_NAME = "om_khangat_tpuf_key"
TURBOPUFFER_API_KEY = "tpuf_dQHBpZEvl612XAdP0MvrQY5dbS0omPMy"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SearchEngine:
    def __init__(self):
        # Initialize Turbopuffer client
        try:
            self.tpuf = turbopuffer.Turbopuffer(
                api_key=TURBOPUFFER_API_KEY,
                region=TURBOPUFFER_REGION,
            )
            self.ns = self.tpuf.namespace(TPUF_NAMESPACE_NAME)
            logging.info("Successfully initialized Turbopuffer client.")
        except Exception as e:
            logging.error(f"Failed to initialize Turbopuffer client: {e}")
            raise RuntimeError("Turbopuffer client setup failed.")

        self.query_rewriter = QueryRewriter()
        self.voyage_client = VoyageAIClient()

    def _get_query_embedding(self, query: str) -> List[float]:
        try:
            return self.voyage_client.embed_query(query)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get query embedding: {e}")

    def search(self, query: str, hard_criteria: dict, soft_criteria: dict) -> List[str]:
        """
        Performs a hybrid search using Turbopuffer: Vector Search -> Post-Filtering (Hard Criteria) -> Re-ranking (Soft Criteria).
        """
        logging.info(f"Processing query: '{query}'")

        # Step 1: Use OpenAI to understand the query and extract attributes
        rewritten_query = self.query_rewriter.rewrite_query(query)
        logging.info(f"OpenAI-rewritten query attributes: {rewritten_query}")

        final_hard_filters = {**hard_criteria, **rewritten_query}

        # Step 2: Vector Search (Semantic Similarity) using Turbopuffer
        try:
            query_vector = self._get_query_embedding(query)
        except RuntimeError as e:
            logging.error(f"Search failed due to embedding error: {e}")
            return []

        # Query Turbopuffer for the top candidates
        # Retrieve more candidates than 'top_k' if you expect many to be filtered out
        try:
            # Build text search filters for full-text search capabilities
            text_filters = []
            if final_hard_filters.get("experience_titles"):
                text_filters.append(f"rerank_summary:{final_hard_filters['experience_titles']}")
            if final_hard_filters.get("education_degrees"):
                text_filters.append(f"rerank_summary:{final_hard_filters['education_degrees']}")
            
            # Combine text filters with OR logic
            text_filter = " OR ".join(text_filters) if text_filters else None
            
            # Perform vector search with optional text filtering
            search_params = {
                "vector": query_vector,
                "top_k": 1000,  # Get many candidates for filtering
                "distance_metric": "cosine_distance"
            }
            
            if text_filter:
                search_params["filters"] = text_filter
                
            response = self.ns.query(**search_params)
            candidates = response.data if hasattr(response, 'data') else response

            logging.info(f"Retrieved {len(candidates)} candidates from Turbopuffer vector search.")

        except Exception as e:
            logging.error(f"Turbopuffer search failed: {e}")
            return []

        # Step 3: Post-Filtering (Hard Criteria) on the retrieved candidates
        # Note: Some filtering might already be done by Turbopuffer's text search
        filtered_candidates_with_meta = []
        
        for candidate in candidates:
            # Extract candidate data from Turbopuffer response
            candidate_id = candidate.get('id')
            similarity_score = 1.0 - candidate.get('dist', 0.0)  # Convert distance to similarity
            
            # Get additional metadata from rerank_summary or other fields
            rerank_summary = candidate.get('rerank_summary', '')
            
            # Apply additional hard filters if needed
            passes_hard_filters = True
            
            # For experience and education filtering, we can check the rerank_summary
            # since it contains bio, experience, and education information
            if final_hard_filters.get("experience_titles"):
                filter_keyword = final_hard_filters["experience_titles"].lower()
                if filter_keyword not in rerank_summary.lower():
                    passes_hard_filters = False

            if passes_hard_filters and final_hard_filters.get("education_degrees"):
                filter_keyword = final_hard_filters["education_degrees"].lower()
                if filter_keyword not in rerank_summary.lower():
                    passes_hard_filters = False

            if passes_hard_filters:
                filtered_candidates_with_meta.append({
                    'id': candidate_id,
                    'vector_score': similarity_score,
                    'metadata': {
                        'name': candidate.get('name', ''),
                        'country': candidate.get('country', ''),
                        'email': candidate.get('email', ''),
                        'linkedin_id': candidate.get('linkedin_id', ''),
                        'rerank_summary': rerank_summary,
                        # Note: We don't have prestige score or years of experience in Turbopuffer
                        # These would need to be added to the migration script if needed
                        'prestigeScore': 0.0,  # Default value
                        'yearsOfWorkExperience': 0.0,  # Default value
                    }
                })

        logging.info(f"Filtered down to {len(filtered_candidates_with_meta)} candidates after hard filtering.")

        # Step 4: Re-ranking (Soft Criteria)
        if not filtered_candidates_with_meta:
            return []  # No candidates left after filtering

        reranked_candidates = []
        for candidate_data in filtered_candidates_with_meta:
            meta = candidate_data['metadata']
            vector_score = candidate_data['vector_score']

            prestige_weight = soft_criteria.get('prestige_score_weight', 0.5)
            experience_weight = soft_criteria.get('experience_weight', 0.5)

            prestige = meta.get('prestigeScore', 0)
            experience = meta.get('yearsOfWorkExperience', 0)

            # Simple re-ranking logic focusing on vector similarity
            # Since we don't have prestige/experience scores from Turbopuffer, we rely more on vector similarity
            final_score = vector_score * 1.0  # Heavy weight on semantic similarity

            reranked_candidates.append({
                'id': candidate_data['id'],
                'final_score': final_score
            })

        reranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return [candidate['id'] for candidate in reranked_candidates[:10]]