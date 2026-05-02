"""
NER Model Module - CRF Model Loading and Inference
Extracted from NLP_SpeedRUN.ipynb

Handles:
- CRF model loading from disk
- Feature extraction for CRF inference
- CRF-based entity extraction
"""

import joblib
import re
import spacy
from typing import List, Dict, Tuple, Any
from pathlib import Path


class CRFFeatureExtractor:
    """Feature extraction for CRF model - matches training pipeline"""
    
    def __init__(self):
        """Initialize spaCy NLP pipeline"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    
    @staticmethod
    def get_word_shape(word: str) -> str:
        """
        Get word shape representation.
        Example: "Alice123" -> "Xxxxddd"
        """
        s = re.sub(r'\d', 'd', word)  # digits -> 'd'
        s = re.sub(r'[A-Z]', 'X', s)  # uppercase -> 'X'
        s = re.sub(r'[a-z]', 'x', s)  # lowercase -> 'x'
        return s
    
    def word2features(self, sent: List[Tuple], i: int) -> Dict[str, Any]:
        """Extract features for word at position i in sentence"""
        word = sent[i][0]
        pos = sent[i][1]
        
        features = {
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word[:2]": word[:2],
            "word[:3]": word[:3],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "word.isalpha()": word.isalpha(),
            "pos": pos,
            "pos[:2]": pos[:2],
            "word.shape": self.get_word_shape(word),
        }
        
        # Previous word features
        if i > 0:
            prev_word = sent[i-1][0]
            prev_pos = sent[i-1][1]
            features.update({
                "-1:word.lower()": prev_word.lower(),
                "-1:word.istitle()": prev_word.istitle(),
                "-1:pos": prev_pos,
            })
        else:
            features["BOS"] = True
        
        # Next word features
        if i < len(sent) - 1:
            next_word = sent[i+1][0]
            next_pos = sent[i+1][1]
            features.update({
                "+1:word.lower()": next_word.lower(),
                "+1:word.istitle()": next_word.istitle(),
                "+1:pos": next_pos,
            })
        else:
            features["EOS"] = True
        
        return features
    
    def sent2features(self, sent: List[Tuple]) -> List[Dict[str, Any]]:
        """Extract all features for a sentence"""
        return [self.word2features(sent, i) for i in range(len(sent))]
    
    def extract_features_from_text(self, text: str) -> Tuple[List[Dict], List[str]]:
        """
        Process raw text and extract CRF features.
        
        Returns:
            Tuple of (features_list, tokens_list)
        """
        doc = self.nlp(text)
        tokens = [t.text for t in doc]
        pos_tags = [t.pos_ for t in doc]
        
        # Build sentence structure with placeholder labels
        sent_for_crf = list(zip(tokens, pos_tags, ['O'] * len(tokens)))
        features = self.sent2features(sent_for_crf)
        
        return features, tokens


class CRFNERModel:
    """CRF-based NER Model wrapper"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize CRF NER Model.
        
        Args:
            model_path: Path to saved CRF model (pkl/joblib file)
                       If None, model must be trained first
        """
        self.model = None
        self.feature_extractor = CRFFeatureExtractor()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained CRF model from disk"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(str(model_path))
        print(f"✅ CRF model loaded from: {model_path}")
    
    def save_model(self, model_path: str) -> None:
        """Save trained CRF model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(model_path))
        print(f"✅ CRF model saved to: {model_path}")
    
    def predict(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text using CRF model.
        
        Args:
            text: Input text
            
        Returns:
            List of entities with format: [{"text": "...", "label": "..."}]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        features, tokens = self.feature_extractor.extract_features_from_text(text)
        
        # CRF predict expects list of sentences
        predictions = self.model.predict([features])[0]
        
        # Extract entities from predictions
        entities = self._extract_entities_from_predictions(tokens, predictions)
        return entities
    
    @staticmethod
    def _extract_entities_from_predictions(
        tokens: List[str], 
        predictions: List[str],
        max_entity_length: int = 30,
        max_entity_tokens: int = None
    ) -> List[Dict[str, str]]:
        """
        Convert token-level predictions to entity spans.
        
        Args:
            tokens: List of tokens
            predictions: List of predicted labels (same length as tokens)
            max_entity_length: Max character length for entity
            max_entity_tokens: Max token count for entity (None = no limit)
        
        Returns:
            List of extracted entities
        """
        results = []
        current_entity = []
        current_label = None
        
        for word, label in zip(tokens, predictions):
            if label == "O":
                if current_entity:
                    entity_text = " ".join(current_entity)
                    # Filters: not full sentence, reasonable length
                    if (len(current_entity) != len(tokens) and 
                        len(entity_text) <= max_entity_length):
                        if max_entity_tokens is None or len(current_entity) <= max_entity_tokens:
                            results.append({
                                "text": entity_text,
                                "label": current_label
                            })
                    current_entity = []
                    current_label = None
                continue
            
            # Entity label (PERSON, ORG, etc.)
            if label != current_label or not current_entity:
                if current_entity:
                    entity_text = " ".join(current_entity)
                    if (len(current_entity) != len(tokens) and 
                        len(entity_text) <= max_entity_length):
                        if max_entity_tokens is None or len(current_entity) <= max_entity_tokens:
                            results.append({
                                "text": entity_text,
                                "label": current_label
                            })
                current_entity = [word]
                current_label = label
            else:
                current_entity.append(word)
        
        # Flush remaining entity
        if current_entity:
            entity_text = " ".join(current_entity)
            if (len(current_entity) != len(tokens) and 
                len(entity_text) <= max_entity_length):
                if max_entity_tokens is None or len(current_entity) <= max_entity_tokens:
                    results.append({
                        "text": entity_text,
                        "label": current_label
                    })
        
        return results
