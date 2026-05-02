import streamlit as st
from ml_models.ner_pipeline import HybridNERPipeline
import os
from pathlib import Path

def ner_main():
    """Main Streamlit application with NER integration"""
    
    st.set_page_config(page_title="NER Pipeline", layout="wide")
    st.title("🔍 Named Entity Recognition Pipeline")
    
    # ===== SIDEBAR: Configuration =====
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path configuration
        use_crf = st.checkbox("Enable CRF Model", value=False)
        crf_model_path = None
        
        if use_crf:
            crf_model_path = st.text_input(
                "CRF Model Path",
                value="./models/crf_ner_model.pkl",
                help="Path to saved CRF model"
            )
        
        st.divider()
        st.markdown("### About")
        st.info(
            "**Hybrid NER Pipeline**\n\n"
            "Combines three extraction methods:\n"
            "1. **Regex**: EMAIL, PHONE, DATE patterns\n"
            "2. **spaCy**: Pre-trained NER model\n"
            "3. **CRF**: Custom trained model (optional)\n\n"
            "Results are merged with priority-based deduplication."
        )
    
    # ===== MAIN CONTENT =====
    
    # Load NER pipeline
    @st.cache_resource
    def load_pipeline(crf_path):
        try:
            return HybridNERPipeline(crf_model_path=crf_path, enable_crf=use_crf)
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            return None
    
    ner_pipeline = load_pipeline(crf_model_path)
    
    if ner_pipeline is None:
        st.stop()
    
    # Text input
    st.subheader("📝 Input Text")
    input_method = st.radio("Choose input method:", ["Text Area", "Upload File"], horizontal=True)
    
    text = ""
    if input_method == "Text Area":
        text = st.text_area(
            "Enter text for NER extraction:",
            height=150,
            placeholder="Type or paste your text here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload text file", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.text_area("File content:", value=text, disabled=True, height=150)
    
    if text:
        # Extract entities
        if st.button("🚀 Extract Entities", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                result = ner_pipeline.extract_with_metadata(text)
            
            # Display results
            st.success("✅ Extraction complete!")
            
            entities = result["entities"]
            stats = result["stats"]
            
            # Results tabs
            tab1, tab2, tab3 = st.tabs(["Entities", "Statistics", "Export"])
            
            with tab1:
                if entities:
                    st.subheader("Found Entities")
                    
                    # Entity table
                    entity_data = []
                    for i, ent in enumerate(entities, 1):
                        entity_data.append({
                            "#": i,
                            "Entity": ent.get("text", ""),
                            "Type": ent.get("label", "MISC"),
                            "Position": ent.get("start", "-")
                        })
                    
                    st.dataframe(entity_data, use_container_width=True, hide_index=True)
                    
                    # Entity breakdown by type
                    st.subheader("Entities by Type")
                    for label in sorted(set(e["label"] for e in entities)):
                        ents_of_type = [e for e in entities if e["label"] == label]
                        with st.expander(f"**{label}** ({len(ents_of_type)})"):
                            for ent in ents_of_type:
                                st.write(f"• {ent['text']}")
                else:
                    st.info("No entities found in the text")
            
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entities", stats["total_entities"])
                with col2:
                    st.metric("Entity Types", len(stats["by_label"]))
                with col3:
                    st.metric("Text Length", len(text))
                
                st.subheader("Entity Distribution")
                if stats["by_label"]:
                    st.bar_chart(stats["by_label"])
                else:
                    st.info("No distribution data")
            
            with tab3:
                st.subheader("Export Results")
                
                import json
                
                # JSON export
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name="ner_results.json",
                    mime="application/json"
                )
                
                # CSV export
                import pandas as pd
                df = pd.DataFrame(entities)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="ner_results.csv",
                    mime="text/csv"
                )