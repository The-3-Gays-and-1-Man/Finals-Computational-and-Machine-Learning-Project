import streamlit as st
from typing import List, Tuple, Dict
from collections import Counter
import plotly.graph_objects as go
import re
from collections import defaultdict


from core.text_processor import TextProcessor
from core.frequency_counter import FrequencyCounter


def render_ner_text(text: str, entities: List[Tuple[int, int, str]]):
    """
    Inline NER renderer with:
    - larger text
    - hover zoom effect
    - entity tooltip
    """

    ENTITY_COLORS = {
        "PERSON": "#FF6B6B",
        "GPE": "#4ECDC4",
        "MISC": "#45B7D1",
        "DATE": "#FFA500",
        "TIME": "#FF6B9D",
        "MONEY": "#95E77D",
        "PERCENT": "#A8E6CF",
        "ORG": "#FFD93D",
        "GPE": "#6BCB77",
        "PRODUCT": "#FF8FB1",
        "EVENT": "#9B59B6",
        "LAW": "#3498DB",
        "LANGUAGE": "#E74C3C",
    }

    # normalize + sort
    spans = []
    for start, end, label in entities:
        try:
            spans.append((int(start), int(end), label))
        except:
            continue

    spans = sorted(spans, key=lambda x: x[0])

    output = ""
    last_idx = 0

    for start, end, label in spans:
        if start > last_idx:
            output += text[last_idx:start]

        color = ENTITY_COLORS.get(label, "#888")
        entity_word = text[start:end]

        output += f"""
        <span class="ner-entity" data-label="{label}" style="
            background-color: {color};
            color: white;
            padding: 10px 15px;
            margin-right: 6px;
            margin-left: 6px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 1.15em;
            display: inline-block;
            transition: all 0.15s ease-in-out;
            cursor: pointer;
        ">
            {entity_word}
        </span>
        """

        last_idx = end

    output += text[last_idx:]

    st.markdown(
        f"""
        <style>
        .ner-entity {{
            position: relative;
        }}

        /* Hover zoom effect */
        .ner-entity:hover {{
            transform: scale(1.08);
            filter: brightness(1.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 10;
        }}

        /* Tooltip */
        .ner-entity:hover::after {{
            content: attr(data-label);
            position: absolute;
            top: -28px;
            left: 50%;
            transform: translateX(-50%);
            background: #222;
            color: white;
            padding: 4px 8px;
            font-size: 0.75em;
            border-radius: 6px;
            white-space: nowrap;
            z-index: 999;
            opacity: 1;
        }}

        /* Smooth text rendering */
        .ner-container {{
            font-size: 1.2em;
            line-height: 2.0;
        }}
        </style>

        <div class="ner-container">
            {output}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_ner_entities(entities: List[tuple], text: str):
    """
    Clean dropdown-style entity viewer (no tables, no dataframe UI)
    """

    st.subheader("🏷️ Detected Entities")

    grouped = defaultdict(list)

    # build groups safely
    for start, end, label in entities:
        try:
            entity_text = text[int(start):int(end)]
            grouped[label].append(entity_text)
        except:
            continue

    # render dropdowns
    for label, items in grouped.items():
        unique_items = sorted(set(items))

        with st.expander(f"{label} ({len(unique_items)})", expanded=False):

            # simple bullet list instead of table
            for item in unique_items[:50]:
                st.markdown(f"- {item}")

            if len(unique_items) > 50:
                st.caption(f"... and {len(unique_items) - 50} more")

def render_ner_text_classification(
    text: str,
    predicted_class: str,
    scores: Dict[str, float],
    category_profiles: Dict,
    overall_profile: Dict
) -> None:
    """
    Render text classification results with confidence scores and explainability.
    
    Args:
        text: Input text
        predicted_class: Predicted category
        scores: Category confidence scores
        category_profiles: Category frequency profiles
        overall_profile: Overall frequency profile
    """
    st.divider()
    st.subheader("🤖 Text Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Result")
        st.success(f"**{predicted_class}** 🎯")
    
    with col2:
        st.subheader("Confidence Scores")
        
        # Create confidence chart
        categories_for_chart = list(scores.keys())
        max_score = max(scores.values()) if scores else 1.0
        min_score = min(scores.values()) if scores else 0.0
        normalized_scores = [
            (scores[cat] - min_score) / (max_score - min_score) * 100
            if max_score != min_score else 50
            for cat in categories_for_chart
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=categories_for_chart, y=normalized_scores, marker_color='#2ca02c')
        ])
        fig.update_layout(height=300, showlegend=False, margin=dict(b=80))
        st.plotly_chart(fig, use_container_width=True)
    
    # Explainability section
    st.subheader("📊 Why This Prediction?")
    
    cat_profile = category_profiles.get(predicted_class)
    if cat_profile:
        processor = TextProcessor()
        tokens_processed = processor.preprocess(text)
        
        cat_counter = cat_profile["counter"]
        cat_total = cat_profile["total"] or 1
        overall_counter = overall_profile["counter"]
        overall_total = overall_profile["total"] or 1
        
        input_counter = Counter(tokens_processed)
        
        evidence = []
        for word, count in input_counter.most_common(20):
            cat_freq = cat_counter.get(word, 0)
            other_freq = overall_counter.get(word, 0) - cat_freq
            other_total = max(overall_total - cat_total, 1)
            
            cat_rel = cat_freq / cat_total
            other_rel = other_freq / other_total
            lift = (cat_rel + 1e-9) / (other_rel + 1e-9)
            
            evidence.append({
                "Word": word,
                "Count in Text": count,
                f"Freq in {predicted_class}%": round(cat_rel * 100, 3),
                "Freq in Others%": round(other_rel * 100, 3),
                "Lift": round(lift, 2)
            })
        
        # Sort by lift then by count
        evidence = sorted(evidence, key=lambda x: (x["Lift"], x["Count in Text"]), reverse=True)
        top_evidence = evidence[:8]
        
        st.info("These words are more common in the predicted category (high lift):")
        st.dataframe(top_evidence, use_container_width=True, hide_index=True)

def render_frequency_analysis(text: str, top_n: int = 20) -> None:
    """
    Render frequency analysis tabs with word statistics and visualizations.
    
    Args:
        text: Input text to analyze
        top_n: Number of top words to display
    """
    st.divider()
    st.subheader("📈 Frequency Analysis")
    
    processor = TextProcessor()
    tokens = processor.preprocess(text, remove_stopwords=True)
    
    counter = FrequencyCounter()
    frequency_map = counter.count_frequencies(tokens)
    top_words = counter.get_top_words(top_n)
    stats = counter.get_statistics()
    
    # Create tabs for analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics",
        "📈 Top Words",
        "📉 Distribution",
        "🔄 Zipf's Law",
        "📖 Advanced Stats"
    ])
    
    # Tab 1: Quick Metrics
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Words", stats['total_words'])
        with col2:
            st.metric("Unique Words", stats['unique_words'])
        with col3:
            st.metric("Type-Token Ratio", f"{stats['type_token_ratio']:.3f}")
        with col4:
            st.metric("Mean Frequency", f"{stats['mean_frequency']:.2f}")
    
    # Tab 2: Bar Chart & Frequency Table
    with tab2:
        st.subheader("Top Words by Frequency")
        if top_words:
            words, freqs = zip(*top_words)
            fig = go.Figure(data=[
                go.Bar(x=list(words), y=list(freqs), marker_color='#1f77b4')
            ])
            fig.update_layout(
                title="Top Words by Frequency",
                xaxis_title="Words",
                yaxis_title="Frequency",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Frequency Table")
            freq_data = []
            for rank, (word, freq) in enumerate(top_words, 1):
                freq_data.append({
                    "Rank": rank,
                    "Word": word,
                    "Frequency": freq,
                    "Percentage": f"{(freq/stats['total_words']*100):.2f}%"
                })
            
            st.dataframe(freq_data, use_container_width=True, hide_index=True)
    
    # Tab 3: Distribution
    with tab3:
        st.subheader("Frequency Distribution")
        all_freqs = [f for _, f in frequency_map.items()]
        fig = go.Figure(data=[
            go.Histogram(x=all_freqs, nbinsx=30, marker_color='#ff7f0e')
        ])
        fig.update_layout(
            title="Frequency Distribution",
            xaxis_title="Frequency",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Zipf's Law
    with tab4:
        st.subheader("Zipf's Law Analysis")
        zipf_data = counter.zipf_law_analysis()
        
        if len(zipf_data) > 0:
            zipf_n = min(top_n, len(zipf_data))
            ranks = [d['rank'] for d in zipf_data[:zipf_n]]
            freqs = [d['frequency'] for d in zipf_data[:zipf_n]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ranks, y=freqs, mode='lines+markers',
                name='Actual', marker_color='#1f77b4'
            ))
            fig.add_trace(go.Scatter(
                x=ranks, y=[1/r for r in ranks],
                mode='lines', name='Zipf Expected',
                line=dict(dash='dash', color='#d62728')
            ))
            
            fig.update_layout(
                title="Zipf's Law Analysis",
                xaxis_title="Rank",
                yaxis_title="Frequency",
                height=400,
                yaxis_type="log",
                xaxis_type="log"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Zipf's Law** states that word frequency follows a power law distribution. "
                     "The dashed line shows the expected distribution if Zipf's Law holds perfectly.")
    
    # Tab 5: Advanced Statistics
    with tab5:
        st.subheader("Comprehensive Statistical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Word Count", stats['total_words'])
            st.metric("Unique Words", stats['unique_words'])
            st.metric("Maximum Frequency", stats['max_frequency'])
        
        with col2:
            st.metric("Mean Frequency", f"{stats['mean_frequency']:.2f}")
            st.metric("Median Frequency", f"{stats['median_frequency']:.2f}")
            st.metric("Min Frequency", stats['min_frequency'])
        
        with col3:
            st.metric("Std Deviation", f"{stats['std_deviation']:.2f}")
            st.metric("Type-Token Ratio", f"{stats['type_token_ratio']:.4f}")
            st.metric("Vocabulary Richness", f"{(stats['type_token_ratio']*100):.2f}%")
        
        st.divider()
        
        # Lexical diversity
        st.subheader("📖 Lexical Diversity Analysis")
        ttr = stats['type_token_ratio']
        
        if ttr < 0.3:
            diversity_level = "Low (Repetitive)"
            diversity_color = "🔴"
            explanation = "Limited vocabulary with many repetitions. May indicate specialized or narrow content."
        elif ttr < 0.6:
            diversity_level = "Medium (Standard)"
            diversity_color = "🟡"
            explanation = "Balanced vocabulary. Typical for most written content."
        else:
            diversity_level = "High (Varied)"
            diversity_color = "🟢"
            explanation = "Rich and diverse vocabulary. Indicates sophisticated or varied content."
        
        st.write(f"{diversity_color} **Diversity Level**: {diversity_level}")
        st.write(f"Type-Token Ratio: **{ttr:.4f}** ({ttr*100:.2f}% unique words)")
        st.write(f"**Interpretation**: {explanation}")

def render_enhanced_ner(
    text: str,
    entities: List[Tuple[int, int, str]],
    predicted_class: str = None,
    scores: Dict[str, float] = None,
    category_profiles: Dict = None,
    overall_profile: Dict = None,
    theme: str = "dark"
) -> None:
    """
    Render complete enhanced NER with visualization, classification, and frequency analysis.
    
    Args:
        text: Input text
        entities: NER entities
        predicted_class: Text classification prediction
        scores: Classification confidence scores
        category_profiles: Category frequency profiles for explainability
        overall_profile: Overall frequency profile
        theme: "dark" or "light"
    """
    # Main NER Visualization
    st.subheader("📝 Named Entity Recognition")
    
    col_viz, col_stats = st.columns([3, 1])
    

    with col_viz:
        render_ner_text(text, entities)

    with col_stats:
        st.subheader("📊 NER Stats")
        st.metric("Characters", len(text))
        st.metric("Words", len(text.split()))
        st.metric("Entities", len(set((s, e, l) for s, e, l in entities)))
    


    render_ner_entities(entities, text)
    
    # Text Classification
    if predicted_class and scores and category_profiles and overall_profile:
        render_ner_text_classification(
            text,
            predicted_class,
            scores,
            category_profiles,
            overall_profile
        )
    
    # Frequency Analysis
    render_frequency_analysis(text, top_n=20)
