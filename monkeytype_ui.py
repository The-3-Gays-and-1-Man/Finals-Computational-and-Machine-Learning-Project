"""
Monkeytype-style UI components for NER visualization
Provides reusable Streamlit components for character-level text rendering with entity highlighting
"""

import streamlit as st
from typing import List, Tuple, Dict, Optional


# Entity color palette
ENTITY_COLORS = {
    "PERSON": "#FF6B6B",
    "LOCATION": "#4ECDC4",
    "ORGANIZATION": "#45B7D1",
    "DATE": "#FFA500",
    "TIME": "#FF6B9D",
    "MONEY": "#95E77D",
    "PERCENT": "#A8E6CF",
    "FACILITY": "#FFD93D",
    "GPE": "#6BCB77",
    "PRODUCT": "#FF8FB1",
    "EVENT": "#9B59B6",
    "LAW": "#3498DB",
    "LANGUAGE": "#E74C3C",
}

# Theme configurations
THEMES = {
    "dark": {
        "background_start": "#1a1a2e",
        "background_end": "#16213e",
        "text_color": "#b0b0b0",
        "cursor_color": "#00ff00",
        "border_color": "#0f3460",
        "text_unentity": "#a0a0a0",
    },
    "light": {
        "background_start": "#f5f5f5",
        "background_end": "#e8e8e8",
        "text_color": "#333333",
        "cursor_color": "#ff0000",
        "border_color": "#cccccc",
        "text_unentity": "#666666",
    },
}


def convert_hex_to_rgb(hex_color: str) -> str:
    """
    Convert hex color (#RRGGBB) to RGB string format (R, G, B).
    
    Args:
        hex_color: Color in hex format, e.g., "#FF6B6B"
    
    Returns:
        RGB string, e.g., "255, 107, 107"
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"


def render_monkeytype_visualization(
    text: str,
    entity_spans: List[Tuple[int, int, str]],
    current_index: int = 0,
    theme: str = "dark"
) -> None:
    """
    Render character-level text visualization with entity highlighting in Monkeytype style.
    
    Features:
    - Character-level HTML rendering
    - Entity color coding
    - Blinking cursor animation
    - Theme support (dark/light)
    - Semi-transparent entity backgrounds
    
    Args:
        text: The input text to visualize
        entity_spans: List of (start, end, entity_label) tuples for entity highlighting
        current_index: Current character index for cursor position (default 0, no cursor)
        theme: "dark" or "light" theme
    """
    theme_config = THEMES.get(theme, THEMES["dark"])
    
    # Build character to entity mapping
    char_to_entity = {}
    for start, end, label in entity_spans:
        for i in range(start, end):
            if i < len(text):
                char_to_entity[i] = label
    
    # Generate HTML with character-level styling
    html = f"""
    <div style="
        background: linear-gradient(135deg, {theme_config['background_start']} 0%, {theme_config['background_end']} 100%);
        padding: 2rem;
        border-radius: 12px;
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        line-height: 1.8;
        color: {theme_config['text_color']};
        border: 2px solid {theme_config['border_color']};
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        word-break: break-word;
        max-width: 100%;
        overflow-x: auto;
    ">
    """
    
    char_idx = 0
    for char in text:
        if char == '\n':
            html += '<br>'
            char_idx += 1
            continue
        
        # Get entity label for this character
        entity_label = char_to_entity.get(char_idx, None)
        entity_color = ENTITY_COLORS.get(entity_label, "#888888") if entity_label else None
        
        # Determine character styling
        if char == ' ':
            # Space character with margin
            html += '<span style="margin-right: 0.25em;"></span>'
        else:
            # Regular or entity character
            if char_idx == current_index:
                # Cursor character (blinking)
                style = f"""
                    color: {theme_config['cursor_color']};
                    background: rgba({convert_hex_to_rgb(theme_config['cursor_color'])}, 0.2);
                    border-bottom: 2px solid {theme_config['cursor_color']};
                    font-weight: bold;
                    animation: blink-cursor 1s infinite;
                """
            elif entity_label and entity_color:
                # Entity character with highlight
                style = f"""
                    color: {entity_color};
                    background: rgba({convert_hex_to_rgb(entity_color)}, 0.15);
                    text-shadow: 0 0 8px rgba({convert_hex_to_rgb(entity_color)}, 0.4);
                    font-weight: 600;
                    padding: 0 2px;
                    border-radius: 2px;
                """
            else:
                # Regular character
                style = f"""
                    color: {theme_config['text_unentity']};
                    opacity: 0.8;
                """
            
            # Escape HTML special characters
            char_display = char.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            html += f'<span style="{style}">{char_display}</span>'
        
        char_idx += 1
    
    html += """
    </div>
    
    <style>
        @keyframes blink-cursor {
            0%, 49% { opacity: 1; }
            50%, 100% { opacity: 0.5; }
        }
    </style>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_entity_legend(
    entity_spans: List[Tuple[int, int, str]],
    text: str
) -> None:
    """
    Render color-coded entity legend with grouped cards.
    
    Features:
    - Groups entities by type
    - Shows unique instances
    - Color-coded left border
    - Semi-transparent backgrounds
    - "+N more" indicator for many instances
    
    Args:
        entity_spans: List of (start, end, entity_label) tuples
        text: Original text for extracting entity strings
    """
    # Group entities by type and collect unique instances
    entities_by_type = {}
    for start, end, label in entity_spans:
        entity_text = text[start:end]
        if label not in entities_by_type:
            entities_by_type[label] = []
        if entity_text not in entities_by_type[label]:
            entities_by_type[label].append(entity_text)
    
    # Render entity cards
    for entity_type in sorted(entities_by_type.keys()):
        texts = entities_by_type[entity_type]
        color = ENTITY_COLORS.get(entity_type, "#888888")
        
        unique_count = len(texts)
        display_texts = texts[:5]
        more_count = max(0, unique_count - 5)
        
        # Build entity list HTML
        entity_list_html = ", ".join([f"<code>{t}</code>" for t in display_texts])
        if more_count > 0:
            entity_list_html += f"<br><em>... and {more_count} more</em>"
        
        # Entity card HTML
        card_html = f"""
        <div style="
            margin: 0.75em 0;
            padding: 1em;
            background: rgba({convert_hex_to_rgb(color)}, 0.08);
            border-left: 4px solid {color};
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        ">
            <div style="flex: 1;">
                <span style="
                    color: {color};
                    font-weight: bold;
                    font-size: 1.1em;
                    display: inline-block;
                    margin-bottom: 0.5em;
                ">{entity_type}</span>
                <div style="
                    font-size: 0.9em;
                    color: #999;
                    margin-top: 0.5em;
                ">
                    {entity_list_html}
                </div>
            </div>
            <div style="
                background: {color};
                color: white;
                padding: 0.25em 0.75em;
                border-radius: 12px;
                font-weight: bold;
                font-size: 0.9em;
                white-space: nowrap;
                margin-left: 1em;
            ">
                {unique_count}
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)


def initialize_ner_session_state() -> None:
    """
    Initialize session state variables for NER functionality.
    
    Creates the following session variables:
    - ner_analyzed: bool - Whether text has been analyzed
    - ner_text_input: str - Current text being analyzed
    - ner_entities: list - Detected entities
    - ner_current_index: int - Current character index (for cursor)
    """
    if 'ner_analyzed' not in st.session_state:
        st.session_state.ner_analyzed = False
    if 'ner_text_input' not in st.session_state:
        st.session_state.ner_text_input = ""
    if 'ner_entities' not in st.session_state:
        st.session_state.ner_entities = []
    if 'ner_current_index' not in st.session_state:
        st.session_state.ner_current_index = 0


def ner_ui_section() -> Tuple[str, List, bool]:
    """
    Render complete NER input UI section with options.
    
    Features:
    - Text input area (500 char limit) or file upload
    - Processing options (stop words, lemmatization, highlighting)
    - Action buttons (Analyze, Reset)
    
    Returns:
        Tuple of (text_input, entities, analyzed_flag)
    """
    initialize_ner_session_state()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio("Input method:", ["Text Input", "Upload File"], horizontal=True)
        
        text_input = ""
        if input_method == "Text Input":
            text_input = st.text_area(
                "Paste your text:",
                height=200,
                placeholder="Enter text for analysis...",
                key="ner_input_area"
            )
            st.caption(f"Characters: {len(text_input)}/500")
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'pdf'])
            if uploaded_file:
                from utils.file_handler import FileHandler
                file_handler = FileHandler()
                text_input = file_handler.read_file(uploaded_file)[:500]
                st.success("✅ File loaded")
    
    with col2:
        st.subheader("Options")
        remove_stopwords = st.checkbox("Remove Stop Words", value=True)
        lemmatize = st.checkbox("Apply Lemmatization", value=False)
        highlight = st.checkbox("Highlight Entities", value=True)
    
    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    return text_input, [], analyze_btn


def render_ner_results(
    text: str,
    entity_spans: List[Tuple[int, int, str]],
    show_legend: bool = True,
    theme: str = "dark"
) -> None:
    """
    Render complete NER results with visualization and entity legend.
    
    Args:
        text: Analyzed text
        entity_spans: Detected entities
        show_legend: Whether to show entity legend
        theme: "dark" or "light"
    """
    col_viz, col_stats = st.columns([3, 1])
    
    with col_viz:
        st.subheader("📝 Text Visualization")
        render_monkeytype_visualization(text, entity_spans, theme=theme)
    
    with col_stats:
        st.subheader("📊 Statistics")
        st.metric("Characters", len(text))
        st.metric("Words", len(text.split()))
        st.metric("Entities Found", len(entity_spans))
    
    if show_legend and entity_spans:
        st.divider()
        st.subheader("🏷️ Entity Reference")
        render_entity_legend(entity_spans, text)
