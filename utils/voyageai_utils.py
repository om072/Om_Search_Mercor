# utils/voyageai_utils.py

import os
import logging
import voyageai
from typing import List

class VoyageAIClient:
    def __init__(self):
        self.api_key = os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            logging.error("VOYAGE_API_KEY not found in environment variables. Embedding will fail.")
            self.client = None
        else:
            self.client = voyageai.Client(api_key=self.api_key)
            logging.info("Voyage AI client initialized for query embedding.")

    def embed_query(self, query: str) -> List[float]:
        """
        Generates a vector embedding for a single text query using the voyage-3 model.
        """
        if not self.client:
            raise RuntimeError("Voyage AI client not initialized due to missing API key.")

        try:
            # The Voyage AI client takes a list of strings
            response = self.client.embed(
                texts=[query],
                model="voyage-3"
            )
            # The API call returns a list of embeddings. We take the first one.
            return response.embeddings[0]
        except Exception as e:
            logging.error(f"Failed to embed query with Voyage AI API: {e}")
            raise RuntimeError(f"Failed to embed query: {e}")