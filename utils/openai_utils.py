import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import Dict, Any

# --- NEW IMPORTS for message types ---
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class QueryRewriter:
    def __init__(self):
        if not OPENAI_API_KEY:
            logging.warning("OPENAI_API_KEY not found. Query rewriting will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            logging.info("OpenAI client initialized for query rewriting.")

    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        Uses a GPT model to extract structured attributes from a natural language query.
        Returns a dictionary of attributes or an empty dict on failure.
        """
        if not self.client:
            return {}

        prompt = f"""
        Extract key search attributes from the following query and format as a JSON object.
        Identify attributes for:
        - job_title (e.g., 'tax lawyer', 'software engineer')
        - required_experience_years (a number, if specified)
        - location (e.g., 'NYC', 'United States')
        - hard_criteria (must-have keywords, e.g., 'corporate law')
        - soft_criteria (nice-to-have keywords, e.g., 'public speaking')
        If an attribute is not present, omit it from the JSON.

        Query: "{query}"
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                # --- CHANGED: Use the instantiated type objects for messages ---
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="You are a helpful assistant that extracts search attributes from a query."
                    ),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=prompt
                    )
                ],
                response_format={"type": "json_object"}
            )
            json_output = response.choices[0].message.content
            return json.loads(json_output)
        except Exception as e:
            logging.error(f"OpenAI API call for query rewriting failed: {e}")
            return {}