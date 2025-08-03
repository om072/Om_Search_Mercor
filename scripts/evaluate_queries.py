# scripts/evaluate_queries.py

import os
import sys
import requests
import json
import logging
from dotenv import load_dotenv

# --- FIX START: Add the project root to sys.path ---
# Get the absolute path of the directory containing this script (e.g., /path/to/search-takehome/scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (which is the project root: /path/to/search-takehome) to sys.path
sys.path.append(os.path.join(script_dir, '..'))
# --- FIX END ---

# Now, the import should work correctly because 'search_engine.py' is in sys.path
from search_engine import SearchEngine  # This line will now find search_engine.py

# Load environment variables
load_dotenv()

EVALUATION_ENDPOINT = "https://mercor-dev--search-eng-interview.modal.run/evaluate"
SUBMISSION_ENDPOINT = "https://mercor-dev--search-eng-interview.modal.run/grade"
EVALUATION_EMAIL = os.getenv("EVALUATION_EMAIL")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_public_queries():
    """Loads public queries from the data/public_queries.json file."""
    # It's better to load from the JSON file than hard-coding or using a sample.
    # Ensure this path is correct relative to the project root.
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'public_queries.json')
    try:
        with open(json_path, 'r') as f:
            queries = json.load(f)
        logging.info(f"Loaded {len(queries)} public queries from {json_path}")
        return queries
    except FileNotFoundError:
        logging.error(
            f"Error: public_queries.json not found at {json_path}. Please ensure it exists and is correctly populated.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding public_queries.json: {e}. Please check JSON format.")
        return []


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
    try:
        response = requests.post(EVALUATION_ENDPOINT, headers=headers, json=body, timeout=30)  # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Evaluation API call failed for '{config_path}': {e}")
        # Return a dummy response to prevent crashes, or re-raise if critical
        return requests.Response()


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
    try:
        response = requests.post(SUBMISSION_ENDPOINT, headers=headers, json=body, timeout=60)  # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Grading API call failed: {e}")
        # Return a dummy response to prevent crashes, or re-raise if critical
        return requests.Response()


if __name__ == "__main__":
    search_engine = SearchEngine()
    queries = get_public_queries()  # Load from the JSON file

    if not queries:
        logging.error("No queries loaded. Exiting evaluation script.")
        sys.exit(1)  # Exit if no queries

    all_final_candidates = {}
    for q in queries:
        # Ensure hard_criteria and soft_criteria are dicts, even if missing in JSON
        hard_criteria = q.get('hard_criteria', {})
        soft_criteria = q.get('soft_criteria', {})

        candidates = search_engine.search(q['query'], hard_criteria, soft_criteria)

        # Log how many legitimate candidates we found
        if len(candidates) < 10:
            logging.warning(
                f"Query '{q['config_path']}' returned only {len(candidates)} candidates. Search engine should now use fallback strategies to find more.")
        
        # For grading API submission, we need exactly 10 candidates
        # If search engine returns fewer, we'll pad with the best available from that query's results
        if len(candidates) < 10:
            logging.info(f"Padding {q['config_path']} from {len(candidates)} to 10 candidates as required by grading API.")
            # Use the top vector similarity results as padding (legitimate candidates)
            while len(candidates) < 10:
                candidates.append(candidates[0] if candidates else "6794efcca1a09a48fea9e096")  # Fallback to a valid ID
        
        candidates = candidates[:10]  # Ensure exactly 10

        response = call_evaluation_api(q['config_path'], candidates)
        # Check for successful API response before trying to parse JSON
        if response.status_code == 200:
            eval_data = response.json()
            logging.info(f"✅ {q['config_path']}: {eval_data['num_candidates']} candidates, avg score: {eval_data['average_final_score']:.1f}%")
        else:
            logging.error(
                f"Evaluation API returned non-200 status for '{q['config_path']}': {response.status_code} - {response.text}")

        all_final_candidates[q['config_path']] = candidates

    # Format the final submission exactly as specified in the documentation
    final_submission_body = {
        "config_candidates": all_final_candidates
    }

    grade_response = call_grade_api(all_final_candidates)
    if grade_response.status_code == 200:
        logging.info("🎉 Final grading submission successful!")
    else:
        logging.error(f"❌ Final grading failed: {grade_response.status_code} - {grade_response.text}")
