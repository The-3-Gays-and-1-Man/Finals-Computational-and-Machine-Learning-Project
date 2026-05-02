import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
from typing import Dict
from core.text_processor import TextProcessor
from core.frequency_counter import FrequencyCounter
from ml_models.naive_bayes import NaiveBayesTextClassifier
from collections import Counter
try:
    from ml_models.sklearn_models import RandomForestText, AVAILABLE as SKLEARN_AVAILABLE
except ImportError:
    RandomForestText = None
    SKLEARN_AVAILABLE = False
from utils.file_handler import FileHandler
from utils.dataset_loader import DatasetLoader
import plotly.graph_objects as go
import plotly.express as px
from ml_models.ner_pipeline import HybridNERPipeline
from app import ner_main
from enhanced_ner import render_enhanced_ner
# Page configuration
st.set_page_config(
    page_title="Word Frequency Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header { font-size: 3em; color: #1f77b4; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 1.5em; border-radius: 10px; }
    .stat-number { font-size: 2em; font-weight: bold; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# Initialize dataset loader and pre-trained models
@st.cache_resource
def load_models_and_data():
    """Load datasets and pre-train baseline and main models on app startup"""
    dataset_loader = DatasetLoader()

    # Prepare combined documents (tokens, label)
    documents = dataset_loader.get_all_documents_combined()

    # Build category frequency profiles for explainability
    category_profiles: Dict[str, Dict] = {}
    overall_counter = Counter()
    for tokens, label in documents:
        cat_counter = category_profiles.setdefault(label, Counter())
        cat_counter.update(tokens)
        overall_counter.update(tokens)

    category_profiles = {
        label: {
            "counter": counter,
            "total": sum(counter.values())
        }
        for label, counter in category_profiles.items()
    }
    overall_profile = {"counter": overall_counter, "total": sum(overall_counter.values())}


    # Train Naive Bayes on ALL datasets combined
    nb_classifier = NaiveBayesTextClassifier()
    if documents:
        nb_classifier.train(documents)
        print(f"Trained Naive Bayes on {len(documents)} documents from all datasets")

    # Train Random Forest on ALL datasets combined (if available)
    rf_classifier = None
    if SKLEARN_AVAILABLE and RandomForestText is not None:
        try:
            texts, labels = dataset_loader.get_all_texts_and_labels_combined()
            if texts and labels:
                rf_classifier = RandomForestText()
                rf_classifier.fit(texts, labels)
                print(f"Trained Random Forest on {len(texts)} samples from all datasets")
        except Exception as e:
            print(f"Random Forest training failed: {e}")
            rf_classifier = None

    return dataset_loader, nb_classifier, rf_classifier, category_profiles, overall_profile

# Load models on startup
dataset_loader, pre_trained_nb, pre_trained_rf, category_profiles, overall_profile = load_models_and_data()

def normalize_entities(entities, text):
    fixed = []

    for e in entities:

        # dict format (your current case)
        if isinstance(e, dict):
            start = e.get("start")
            label = e.get("label")

            # IMPORTANT FIX: reconstruct end safely
            text_value = e.get("text", "")

            if start is not None and label:
                end = e.get("end")
                if end is None:
                    end = start + len(text_value)

                fixed.append((int(start), int(end), label))

        # tuple format (future-proof)
        elif isinstance(e, tuple) and len(e) == 3:
            fixed.append(e)

    return fixed

def main():

    # Initialize session state
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    if 'tokens' not in st.session_state:
        st.session_state.tokens = []

    ner_main()

def ner_main():
    """Main Streamlit app with Hybrid NER + Classification + Frequency Analysis"""

    st.set_page_config(page_title="NER Pipeline", layout="wide")
    st.title("🔍 Named Entity Recognition Pipeline")

    enable_classification= True
    enable_frequency = True

    # ================= LOAD PIPELINE =================
    @st.cache_resource
    def load_pipeline(crf_path, use_crf_flag):
        return HybridNERPipeline(
            crf_model_path=crf_path,
            enable_crf=use_crf_flag
        )

    ner_pipeline = load_pipeline(
        "./models/crf_ner_model.pkl",
        True
    )

    if ner_pipeline is None:
        st.stop()

    # ================= SESSION STATE =================
    if "ner_text_input" not in st.session_state:
        st.session_state.ner_text_input = ""
    if "ner_entities" not in st.session_state:
        st.session_state.ner_entities = []
    if "ner_classification" not in st.session_state:
        st.session_state.ner_classification = None
    if "ner_analyzed" not in st.session_state:
        st.session_state.ner_analyzed = False

    # ================= INPUT =================
    st.subheader("📝 Input Text")

    text_input = st.text_area(
        "Enter text:",
        height=150,
        placeholder="Type or paste text here..."
    )

    # ================= ANALYZE BUTTON =================
    if st.button("🚀 Analyze", type="primary", use_container_width=True):

        if not text_input.strip():
            st.warning("Please enter text first")
            return

        with st.spinner("Running Hybrid NER pipeline..."):

            # ---- NER ----
            result = ner_pipeline.extract_with_metadata(text_input)
            entities = normalize_entities(result.get("entities", []), text_input)

            st.session_state.ner_entities = entities
            st.session_state.ner_text_input = text_input
            st.session_state.ner_analyzed = True

            # ---- CLASSIFICATION ----
            if enable_classification:
                processor = TextProcessor()
                tokens = processor.preprocess(text_input)
                predicted_class, scores = pre_trained_nb.predict(tokens)

                st.session_state.ner_classification = {
                    "predicted_class": predicted_class,
                    "scores": scores
                }
            else:
                st.session_state.ner_classification = None

        st.success("✅ Analysis complete!")

    # ================= OUTPUT =================
    if st.session_state.ner_analyzed and st.session_state.ner_text_input:

        st.divider()

        classification_data = st.session_state.ner_classification

        render_enhanced_ner(
            text=st.session_state.ner_text_input,
            entities=st.session_state.ner_entities,
            predicted_class=classification_data["predicted_class"] if classification_data else None,
            scores=classification_data["scores"] if classification_data else None,
            category_profiles=category_profiles if enable_frequency else None,
            overall_profile=overall_profile if enable_frequency else None,
        )


if __name__ == "__main__":
    main()
