"""
NER Pipeline Integration Module
Formats and displays entity extraction results
"""

from typing import List, Dict
from ml_models.ner_pipeline import HybridNERPipeline


def format_entities_display(entities: List[Dict]) -> str:
    """
    Format entities in the desired display format:
    
    📌 ENTITIES FOUND:
    
    • Entity Text       → LABEL
    • Another Entity    → LABEL
    
    Args:
        entities: List of dicts with 'text' and 'label' keys
        
    Returns:
        Formatted string ready for display
    """
    if not entities:
        return "❌ No entities found"
    
    output = "📌 **ENTITIES FOUND:**\n\n"
    for ent in entities:
        text = ent.get("text", "").strip()
        label = ent.get("label", "MISC")
        # Format: • Text → LABEL (with padding for alignment)
        output += f"• {text:<30} → {label}\n"
    
    return output


def format_entities_by_type(entities: List[Dict]) -> str:
    """Format entities grouped by type"""
    if not entities:
        return "No entities found"
    
    by_type = {}
    for ent in entities:
        label = ent.get("label", "MISC")
        if label not in by_type:
            by_type[label] = []
        by_type[label].append(ent.get("text", ""))
    
    output = "📌 **ENTITIES BY TYPE:**\n\n"
    for label in sorted(by_type.keys()):
        items = by_type[label]
        output += f"**{label}** ({len(items)}):\n"
        for item in items:
            output += f"  • {item}\n"
        output += "\n"
    
    return output


def generate_entity_html(text: str, entities: List[Dict]) -> str:
    """Generate highlighted HTML visualization of text with entities"""
    # Build character-to-entity mapping
    char_to_entity = {}
    for ent in entities:
        label = ent.get("label", "MISC")
        start = ent.get("start", 0)
        entity_text = ent.get("text", "")
        for i in range(len(entity_text)):
            if start + i < len(text):
                char_to_entity[start + i] = label
    
    # Color palette for entity types
    colors = {
        "PERSON": "#FF6B6B",
        "LOCATION": "#4ECDC4",
        "GPE": "#6BCB77",
        "ORG": "#45B7D1",
        "ORGANIZATION": "#45B7D1",
        "DATE": "#FFA500",
        "TIME": "#FF6B9D",
        "MONEY": "#95E77D",
        "EMAIL": "#9B59B6",
        "PHONE": "#16A085",
        "MISC": "#888888",
    }
    
    # Build HTML with styling
    html = '<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2rem; border-radius: 12px; font-family: \'Courier New\', monospace; font-size: 1.1em; line-height: 1.8; color: #b0b0b0; border: 2px solid #0f3460; word-break: break-word;">'
    
    for char_idx, char in enumerate(text):
        if char == '\n':
            html += '<br>'
        elif char == ' ':
            html += ' '
        else:
            label = char_to_entity.get(char_idx)
            color = colors.get(label, "#888888") if label else "#888888"
            
            if label:
                style = f'color: {color}; font-weight: 600; text-shadow: 0 0 8px rgba(155, 89, 182, 0.4);'
            else:
                style = 'color: #a0a0a0; opacity: 0.7;'
            
            html += f'<span style="{style}">{char}</span>'
    
    html += '</div>'
    return html
