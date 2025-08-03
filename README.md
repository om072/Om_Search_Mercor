# LinkedIn Profile Search Engine

A sophisticated search engine for LinkedIn profiles that combines vector similarity search with advanced filtering and ranking capabilities. This system uses FAISS for efficient vector search, MongoDB for data storage, and AI-powered query processing to find the most relevant LinkedIn profiles based on complex search criteria.

## 🚀 Features

- **Vector Similarity Search**: Uses FAISS for fast and accurate semantic search
- **Advanced Filtering**: Support for hard criteria (must-have) and soft criteria (nice-to-have)
- **AI-Powered Query Processing**: Leverages OpenAI for query understanding and rewriting
- **Hybrid Search Pipeline**: Combines vector search with post-filtering and re-ranking
- **Scalable Architecture**: Batch processing for large datasets
- **Comprehensive Metadata**: Tracks experience, education, skills, and more

## 🏗️ Architecture

The system consists of several key components:

- **Search Engine** (`search_engine.py`): Main search functionality with hybrid filtering
- **Initialization** (`init.py`): Data ingestion and FAISS index building
- **Utilities**: OpenAI integration, VoyageAI embeddings, and string processing
- **Pipeline Scripts**: Automated setup and evaluation workflows

## 📋 Prerequisites

- Python 3.8+
- MongoDB instance
- API keys for:
  - OpenAI
  - VoyageAI

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with your API keys:
   ```env
   MONGO_URI=your_mongodb_connection_string
   OPENAI_API_KEY=your_openai_api_key
   VOYAGEAI_API_KEY=your_voyageai_api_key
   ```

## 🚀 Quick Start

### Option 1: Automated Pipeline
Run the complete pipeline with a single command:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Option 2: Manual Setup

1. **Initialize the search index**
   ```bash
   python init.py
   ```

2. **Run search queries**
   ```python
   from search_engine import SearchEngine
   
   # Initialize the search engine
   se = SearchEngine()
   
   # Perform a search
   results = se.search(
       query="software engineer with 5+ years experience in Python",
       hard_criteria={"yearsOfWorkExperience": 5},
       soft_criteria={"skills": "python"}
   )
   ```

## 📁 Project Structure

```
├── search_engine.py          # Main search engine implementation
├── init.py                   # Data ingestion and index building
├── requirements.txt          # Python dependencies
├── run_pipeline.sh          # Automated pipeline script
├── utils/                   # Utility modules
│   ├── openai_utils.py     # OpenAI integration
│   ├── voyageai_utils.py   # VoyageAI embeddings
│   └── string_utils.py     # String processing utilities
├── scripts/                 # Evaluation and utility scripts
├── data/                    # Data storage directory
└── venv/                    # Virtual environment
```

## 🔧 Configuration

### MongoDB Setup
- Database: `interview_data`
- Collection: `linkedin_data_subset`
- Ensure your MongoDB instance contains LinkedIn profile data with embeddings

### FAISS Index Files
The system creates and uses several index files:
- `linkedin_profiles.faiss`: FAISS vector index
- `linkedin_ids_to_index_mapping.pkl`: Index to MongoDB ID mapping
- `linkedin_index_to_ids.pkl`: Metadata mapping

## 🎯 Usage Examples

### Basic Search
```python
from search_engine import SearchEngine

se = SearchEngine()

# Simple search
results = se.search(
    query="machine learning engineer",
    hard_criteria={},
    soft_criteria={}
)
```

### Advanced Search with Filters
```python
# Search with specific criteria
results = se.search(
    query="senior software engineer",
    hard_criteria={
        "yearsOfWorkExperience": 5,
        "country": "United States"
    },
    soft_criteria={
        "skills": "python,javascript",
        "experience_titles": "software engineer,developer"
    }
)
```

## 🔍 Search Features

### Hard Criteria (Must-Have)
- `yearsOfWorkExperience`: Minimum years of work experience
- `country`: Specific country requirement
- `experience_titles`: Required job titles in experience

### Soft Criteria (Nice-to-Have)
- `skills`: Preferred skills (comma-separated)
- `education_degrees`: Preferred degrees
- `education_fields`: Preferred fields of study

### Query Processing
The system automatically:
1. Rewrites queries using OpenAI for better understanding
2. Extracts relevant attributes from natural language
3. Combines with explicit criteria for comprehensive search

## 📊 Performance

- **Vector Search**: FAISS provides sub-second search times
- **Batch Processing**: Efficient handling of large datasets
- **Memory Efficient**: Streaming processing for large document collections
- **Scalable**: Supports millions of profiles

## 🧪 Testing

Run the test suite:
```bash
python test.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include error logs and reproduction steps

## 🔄 Updates

- **v1.0**: Initial release with FAISS-based search
- **v1.1**: Added hybrid filtering and AI query processing
- **v1.2**: Improved batch processing and performance optimizations

---

**Note**: This system requires a properly configured MongoDB instance with LinkedIn profile data and appropriate API keys for full functionality. 
