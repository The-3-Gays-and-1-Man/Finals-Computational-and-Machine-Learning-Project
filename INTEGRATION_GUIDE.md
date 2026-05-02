# NER Pipeline Integration - Complete Setup Guide

## 📋 Project Structure

```
Finals-Computational-and-Machine-Learning-Project/
├── ml_models/
│   ├── __init__.py
│   ├── ner_model.py              # CRF model loading & inference
│   ├── ner_extractors.py         # Regex & spaCy extractors
│   └── ner_pipeline.py           # Hybrid merger engine
├── models/
│   └── crf_ner_model.pkl         # Trained CRF model (from notebook)
├── app.py                        # Streamlit application
├── enhanced_ner.py               # Streamlit UI components
├── utils.py                      # Utilities (already created ✓)
├── requirements.txt              # Dependencies
└── INTEGRATION_GUIDE.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Export Model from Notebook
From your Colab notebook, add this cell at the end:

```python
import joblib
from pathlib import Path

# Create models directory
Path('./models').mkdir(exist_ok=True)

# Save trained CRF model
model_path = './models/crf_ner_model.pkl'
joblib.dump(crf, model_path)
print(f'✅ Model saved to: {model_path}')

# Download to local machine
from google.colab import files
files.download(model_path)
```

Place the downloaded file in `./models/crf_ner_model.pkl`

### 3. Run Streamlit App
```bash
streamlit run app.py
```

## 📦 Module Reference

### ner_model.py - CRF Model
```python
from ml_models.ner_model import CRFNERModel

# Load pretrained model
crf_model = CRFNERModel(model_path='./models/crf_ner_model.pkl')

# Extract entities
entities = crf_model.predict("John Smith works at Google")
# Returns: [{"text": "John Smith", "label": "PERSON"}, ...]
```

**Features:**
- Feature extraction (word shape, POS, n-grams)
- Token-level predictions
- Entity span extraction
- Model persistence (joblib)

### ner_extractors.py - Rule-Based & Pre-trained
```python
from ml_models.ner_extractors import RegexEntityExtractor, SpacyEntityExtractor

# Regex patterns
regex = RegexEntityExtractor()
emails = regex.extract("Contact: john@gmail.com")

# SpaCy NER
spacy_ner = SpacyEntityExtractor()
entities = spacy_ner.extract("Apple Inc. is in Cupertino")
```

**Supported Patterns:**
- EMAIL: standard email format
- PHONE: international phone numbers
- DATE: multiple date formats
- PERSON, ORG, GPE, DATE (spaCy)

### ner_pipeline.py - Hybrid Merger
```python
from ml_models.ner_pipeline import HybridNERPipeline

# Initialize (CRF is optional)
ner = HybridNERPipeline(
    crf_model_path='./models/crf_ner_model.pkl',
    enable_crf=True
)

# Extract with metadata
result = ner.extract_with_metadata("Your text here")

# Returns:
# {
#   "text": "...",
#   "entities": [{"text": "...", "label": "...", "start": ...}],
#   "stats": {"total_entities": N, "by_label": {...}}
# }
```

**Merging Strategy:**
- Priority scoring: EMAIL/PHONE (10) > DATE (5) > entities (1-4)
- Deduplicates by normalized text
- Filters redundant substrings
- Maintains source order

### enhanced_ner.py - Streamlit UI
```python
from enhanced_ner import render_enhanced_ner

render_enhanced_ner(
    text="Your text",
    ner_pipeline=pipeline,
    crf_model_path="./models/crf_ner_model.pkl"
)
```

**Features:**
- Tabbed interface (Entities, Analysis, Details)
- Entity highlighting with color coding
- Statistics visualization
- Type distribution charts

### utils.py - Utilities (Already Created ✓)

**ModelManager:**
```python
from utils import ModelManager

mm = ModelManager("./models")
mm.save_model(trained_model, "my_model", overwrite=False)
loaded = mm.load_model("my_model")
mm.list_models()  # ["crf_ner_model", "my_model"]
```

**TextProcessor:**
```python
from utils import TextProcessor

clean = TextProcessor.clean_text("messy  text\n\n")
sentences = TextProcessor.split_sentences("Hello. World!")
truncated = TextProcessor.truncate_text(long_text, max_length=512)
```

**EvaluationMetrics:**
```python
from utils import EvaluationMetrics

metrics = EvaluationMetrics.compute_entity_match(true_ents, pred_ents)
# Returns: {"precision": 0.9, "recall": 0.85, "f1": 0.87, ...}

overlap = EvaluationMetrics.compute_partial_match(true_ents, pred_ents)
```

**Batch Processing:**
```python
from utils import batch_extract, export_results

texts = ["Text 1", "Text 2", "Text 3"]
results = batch_extract(ner_pipeline, texts, batch_size=32)

export_results(results, "./output/results.json", format="json")
export_results(results, "./output/results.jsonl", format="jsonl")
export_results(results, "./output/results.csv", format="csv")
```

## 🔄 Integration Examples

### Minimal Script
```python
from ml_models.ner_pipeline import HybridNERPipeline

ner = HybridNERPipeline(crf_model_path="./models/crf_ner_model.pkl")
text = "John Smith works at Google. Email: john@gmail.com"
entities = ner.extract(text)

for ent in entities:
    print(f"{ent['text']} -> {ent['label']}")
```

### FastAPI Endpoint
```python
from fastapi import FastAPI
from pydantic import BaseModel
from ml_models.ner_pipeline import HybridNERPipeline

app = FastAPI()
ner = HybridNERPipeline(crf_model_path="./models/crf_ner_model.pkl")

class TextRequest(BaseModel):
    text: str

@app.post("/extract")
def extract_entities(request: TextRequest):
    result = ner.extract_with_metadata(request.text)
    return result

# Run: uvicorn your_file:app --reload
```

### Batch Processing
```python
from utils import batch_extract, export_results

texts = [
    "John Smith works at Google",
    "Contact Alice Brown at alice@company.com",
    "Meeting on 2024-01-15"
]

results = batch_extract(ner, texts)
export_results(results, "./results.json", format="json")
```

## ⚙️ Configuration

### Model Loading Priority
1. **If CRF model exists**: Uses all 3 methods (Regex + spaCy + CRF)
2. **If CRF missing**: Falls back to Regex + spaCy only
3. **Graceful degradation**: App continues even if spaCy fails

### Environment Setup
Create `.env` file (optional):
```
CRF_MODEL_PATH=./models/crf_ner_model.pkl
ENABLE_CRF=true
SPACY_MODEL=en_core_web_sm
```

## 🎯 Performance Notes

| Component | Latency | Memory |
|-----------|---------|--------|
| Regex     | ~1ms    | Minimal |
| spaCy     | ~50ms   | ~100MB |
| CRF       | ~100ms  | ~50MB  |
| **Total** | ~150ms  | ~150MB |

**Optimization Tips:**
- Cache pipeline in Streamlit using `@st.cache_resource`
- Batch process texts when possible
- Use `enable_crf=False` for faster inference (Regex+spaCy only)

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| spaCy model not found | `python -m spacy download en_core_web_sm` |
| CRF model not found | Export from notebook: `joblib.dump(crf, path)` |
| Import errors | `pip install -r requirements.txt` |
| Streamlit cache issues | Clear cache: `streamlit cache clear` |

## 📊 What Was Refactored

### From Notebook ✗
- Global variables
- Sequential cells
- Hard-coded paths
- Manual feature engineering
- Monolithic pipeline

### To Production Code ✓
- Modular classes
- Reusable functions
- Configuration management
- Abstracted feature extraction
- Pluggable extractors

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
# Accessible at http://localhost:8501
```

### Production (FastAPI)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## 📝 Files to Create/Update

✓ `ml_models/ner_model.py` - Created
✓ `ml_models/ner_extractors.py` - Created
✓ `ml_models/ner_pipeline.py` - Created
✓ `enhanced_ner.py` - Created
✓ `utils.py` - Already provided
⏳ `app.py` - Ready to create
⏳ `requirements.txt` - Ready to create

## 🎓 Key Concepts

**Feature Extraction (CRF):**
- Word shape: "Alice123" → "Xxxddd"
- Contextual features: previous/next words, POS tags
- Morphological: uppercase, digit, alpha flags

**Priority-Based Merging:**
1. Sort all entities by (label_priority, text_length)
2. Deduplicate by normalized text
3. Filter redundant substrings
4. Return sorted by position

**Model Persistence:**
- Saved: `joblib.dump(model, path)`
- Loaded: `joblib.load(path)`
- Size: ~50-100MB typical

---

**Next Steps:**
1. Download trained CRF model from Colab
2. Place in `./models/crf_ner_model.pkl`
3. Run `pip install -r requirements.txt`
4. Launch `streamlit run app.py`
