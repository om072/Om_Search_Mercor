import os
import requests
import json
import logging
from dotenv import load_dotenv
from search_engine import SearchEngine

# Load environment variables
load_dotenv()

EVALUATION_ENDPOINT = "https://mercor-dev--search-eng-interview.modal.run/evaluate"
SUBMISSION_ENDPOINT = "https://mercor-dev--search-eng-interview.modal.run/grade"
EVALUATION_EMAIL = os.getenv("EVALUATION_EMAIL")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_public_queries():
    """Loads public queries from a predefined structure."""
    # This is where you would load the queries from the public spreadsheet.
    # For simplicity, we'll use a sample. You need to expand this.
    return [
        {
            "config_path": "tax_lawyer.yml",
            "query": "experienced tax lawyer with at least 5 years of experience",
            "hard_criteria": {"yearsOfWorkExperience": 5},
            "soft_criteria": {"prestige_score_weight": 0.7, "experience_weight": 0.3}
        },
        # Add all other queries here, with their hard/soft criteria.
    ]

def call_evaluation_api(config_path: str, object_ids: list[str]) -> requests.Response:
    """Calls the Mercor evaluation endpoint."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': EVALUATION_EMAIL,
    }
    body = {
        "config_path": config_path,
        "object_ids": object_ids
    }
    logging.info(f"Submitting candidates for '{config_path}' to evaluation endpoint.")
    return requests.post(EVALUATION_ENDPOINT, headers=headers, json=body)

def call_grade_api(config_candidates: dict) -> requests.Response:
    """Calls the Mercor final grading endpoint."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': EVALUATION_EMAIL,
    }
    body = {
        "config_candidates": config_candidates
    }
    logging.info("Submitting final candidates for grading.")
    return requests.post(SUBMISSION_ENDPOINT, headers=headers, json=body)


if __name__ == "__main__":
    search_engine = SearchEngine()
    queries = get_public_queries()
    
    all_final_candidates = {}
    for q in queries:
        candidates = search_engine.search(q['query'], q['hard_criteria'], q['soft_criteria'])
        
        response = call_evaluation_api(q['config_path'], candidates)
        logging.info(f"Evaluation response for '{q['config_path']}': {response.json()}")

        all_final_candidates[q['config_path']] = candidates
    
    final_submission_body = {q['config_path']: all_final_candidates[q['config_path']][:10] for q in queries}
    
    grade_response = call_grade_api(final_submission_body)
    logging.info(f"Final submission response: {grade_response.json()}")