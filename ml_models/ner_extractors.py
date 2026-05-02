"""
NER Extraction Modules - Regex and spaCy based extractors
Extracted from NLP_SpeedRUN.ipynb

Provides three independent entity extraction methods:
1. Regex-based (EMAIL, PHONE, DATE patterns)
2. spaCy NER (pre-trained model)
3. CRF-based (custom trained model)
"""

import re
import spacy
from typing import List, Dict
from pathlib import Path


class RegexEntityExtractor:
    """Rule-based entity extraction using regex patterns"""
    
    def __init__(self):
        """Initialize regex patterns for common entity types"""
        self.patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s-]?\d{3,4}[\s-]?\d{3,9}",
            "DATE": r"\b(?:(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-]\d{4}|\d{4}-\d{2}-\d{2})\b"
        }
    
    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of entities: [{"text": "...", "label": "...", "start": ..., "end": ...}]
        """
        results = []
        
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                results.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })
        
        return results


class SpacyEntityExtractor:
    """spaCy pre-trained NER extractor"""
    
    def __init__(self):
        """Initialize spaCy NLP pipeline"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    
    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            List of entities: [{"text": "...", "label": "...", "start": ..., "end": ...}]
        """
        doc = self.nlp(text)
        results = []
        
        for ent in doc.ents:
            # Normalize spaCy labels to our schema
            label = ent.label_
            
            # Map spaCy labels to our categories
            if label not in ["PERSON", "ORG", "GPE", "DATE"]:
                label = "MISC"
            
            results.append({
                "text": ent.text,
                "text_lower": ent.text.lower(),
                "label": label,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return results
