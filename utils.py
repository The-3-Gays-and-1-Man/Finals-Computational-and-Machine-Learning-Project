"""
Utility functions for NER pipeline
Helpers for model management, text processing, and evaluation
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import joblib


class ModelManager:
    """Handles model loading, saving, and version management"""
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str, model_type: str = "pkl") -> Path:
        """Get full path for a model file"""
        return self.model_dir / f"{model_name}.{model_type}"
    
    def save_model(self, model, model_name: str, overwrite: bool = False) -> Path:
        """
        Save model to disk using joblib.
        
        Args:
            model: Model object to save
            model_name: Name for the model file
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved model
        """
        model_path = self.get_model_path(model_name)
        
        if model_path.exists() and not overwrite:
            raise FileExistsError(
                f"Model already exists: {model_path}. Use overwrite=True to replace."
            )
        
        joblib.dump(model, str(model_path))
        print(f"✅ Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_name: str) -> object:
        """Load model from disk"""
        model_path = self.get_model_path(model_name)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return joblib.load(str(model_path))
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return [f.stem for f in self.model_dir.glob("*")]
    
    def save_metadata(self, model_name: str, metadata: Dict) -> Path:
        """Save model metadata as JSON"""
        meta_path = self.model_dir / f"{model_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return meta_path
    
    def load_metadata(self, model_name: str) -> Dict:
        """Load model metadata"""
        meta_path = self.model_dir / f"{model_name}_metadata.json"
        
        if not meta_path.exists():
            return {}
        
        with open(meta_path) as f:
            return json.load(f)


class TextProcessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove special control characters
        text = "".join(ch for ch in text if ch.isprintable() or ch in '\n\t')
        return text
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences (simple approach).
        For production, use nltk.sent_tokenize or spaCy.
        """
        import re
        # Split on period, question mark, exclamation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate text to max length"""
        return text[:max_length]


class EvaluationMetrics:
    """Metrics for evaluating NER predictions"""
    
    @staticmethod
    def compute_entity_match(true_ents: List[Dict], pred_ents: List[Dict]) -> Dict:
        """
        Compute exact match metrics between true and predicted entities.
        
        Args:
            true_ents: Ground truth entities [{"text": "...", "label": "..."}]
            pred_ents: Predicted entities
            
        Returns:
            Dict with precision, recall, f1
        """
        true_set = set((e.get("text"), e.get("label")) for e in true_ents)
        pred_set = set((e.get("text"), e.get("label")) for e in pred_ents)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    @staticmethod
    def compute_partial_match(true_ents: List[Dict], pred_ents: List[Dict]) -> Dict:
        """
        Compute partial match metrics (entity boundaries can overlap).
        Useful for token-level evaluation.
        """
        # Extract spans
        true_spans = set((e.get("start"), e.get("start", 0) + len(e.get("text", ""))) 
                        for e in true_ents if "start" in e)
        pred_spans = set((e.get("start"), e.get("start", 0) + len(e.get("text", ""))) 
                        for e in pred_ents if "start" in e)
        
        if not true_spans or not pred_spans:
            return {"overlap": 0, "partial_precision": 0, "partial_recall": 0}
        
        # Count overlaps
        overlaps = 0
        for true_start, true_end in true_spans:
            for pred_start, pred_end in pred_spans:
                # Check if spans overlap
                if pred_start < true_end and pred_end > true_start:
                    overlaps += 1
                    break
        
        pp = overlaps / len(pred_spans) if pred_spans else 0
        pr = overlaps / len(true_spans) if true_spans else 0
        
        return {
            "overlapping_entities": overlaps,
            "partial_precision": pp,
            "partial_recall": pr
        }


def batch_extract(ner_pipeline, texts: List[str], batch_size: int = 32) -> List[Dict]:
    """
    Extract entities from multiple texts efficiently.
    
    Args:
        ner_pipeline: HybridNERPipeline instance
        texts: List of texts
        batch_size: Batch size for processing
        
    Returns:
        List of results [{"text": "...", "entities": [...], "stats": {...}}]
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            try:
                result = ner_pipeline.extract_with_metadata(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "text": text,
                    "entities": [],
                    "error": str(e)
                })
    
    return results


def export_results(results: List[Dict], output_path: str, format: str = "json"):
    """
    Export NER results to file.
    
    Args:
        results: List of NER results
        output_path: Where to save
        format: "json", "jsonl", or "csv"
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif format == "jsonl":
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    
    elif format == "csv":
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Text", "Entity", "Label", "Position"])
            
            for result in results:
                text = result.get("text", "")
                for entity in result.get("entities", []):
                    writer.writerow([
                        text[:100],  # Truncate text for CSV
                        entity.get("text", ""),
                        entity.get("label", ""),
                        entity.get("start", "")
                    ])
    
    print(f"✅ Results exported to: {output_path}")
