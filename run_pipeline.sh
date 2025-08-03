#!/bin/bash

# run_pipeline.sh
# This script automates the entire search pipeline, from environment setup to evaluation.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting the automated search pipeline..."
echo "----------------------------------------"

# --- Step 1: Check Environment and Install Dependencies ---
echo "1. Checking for .env file and installing dependencies..."

if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one with your API keys."
    exit 1
fi

# Activate a virtual environment (optional but recommended)
# python3 -m venv venv
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

echo "Dependencies installed successfully."
echo "----------------------------------------"

# --- Step 2: Data Ingestion and Indexing ---
# This step loads data from MongoDB and indexes it into Turbopuffer.
echo "2. Running data ingestion and indexing (init.py)..."
python init.py

echo "Data ingestion and indexing completed."
echo "----------------------------------------"

# --- Step 3: Run Search and Evaluation ---
# This step runs your search logic against the public queries and submits for evaluation.
echo "3. Running search and evaluation (evaluate_queries.py)..."
python scripts/evaluate_queries.py

echo "Search and evaluation completed."
echo "----------------------------------------"

echo "Pipeline finished successfully."

# Deactivate virtual environment (if used)
# deactivate