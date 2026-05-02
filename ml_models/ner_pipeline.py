"""
Hybrid NER Pipeline - Merges multiple extraction methods
Extracted from NLP_SpeedRUN.ipynb

Core innovation: Priority-based entity deduplication and merging
"""

from typing import List, Dict
from ml_models.ner_model import CRFNERModel
from ml_models.ner_extractors import RegexEntityExtractor, SpacyEntityExtractor


class HybridNERPipeline:
    """
    Hybrid NER combining three extraction methods with intelligent merging.
    
    Priority order:
    1. Regex (EMAIL, PHONE) - highest precision patterns
    2. DATE patterns
    3. spaCy NER & CRF (general entities)
    """
    
    # Priority scoring for deduplication
    LABEL_PRIORITY = {
        "EMAIL": 10,
        "PHONE": 10,
        "DATE": 5,
        "PERSON": 4,
        "ORG": 4,
        "GPE": 4,
        "PRODUCT": 2,
        "MISC": 1
    }
    
    def __init__(self, crf_model_path: str = None, enable_crf: bool = True):
        """
        Initialize hybrid NER pipeline.
        
        Args:
            crf_model_path: Path to trained CRF model (optional)
            enable_crf: Whether to use CRF extractor (requires model_path if True)
        """
        self.regex_extractor = RegexEntityExtractor()
        self.spacy_extractor = SpacyEntityExtractor()
        self.crf_model = None
        self.enable_crf = enable_crf
        
        if enable_crf and crf_model_path:
            try:
                self.crf_model = CRFNERModel(model_path=crf_model_path)
                print("✅ CRF model loaded successfully")
            except Exception as e:
                print(f"⚠️  CRF model failed to load: {e}. Falling back to regex + spaCy only")
                self.crf_model = None
                self.enable_crf = False
    
    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities using all available methods and merge results.
        
        Args:
            text: Input text
            
        Returns:
            List of merged entities: [{"text": "...", "label": "...", "start": ...}]
        """
        # Extract from all methods
        regex_results = self.regex_extractor.extract(text)
        spacy_results = self.spacy_extractor.extract(text)
        crf_results = self.crf_model.predict(text) if self.crf_model else []
        
        # Merge results
        final_entities = self._merge_entities(
            regex_results,
            spacy_results,
            crf_results
        )
        
        return final_entities
    
    def _merge_entities(self, *entity_sources: List[Dict]) -> List[Dict]:
        """
        Merge entity results from multiple sources with intelligent deduplication.
        
        Strategy:
        1. Prioritize by label importance (EMAIL/PHONE > DATE > general entities)
        2. Filter by length (priority to longer/more specific spans)
        3. Remove redundant substrings
        4. Deduplicate normalized text
        
        Args:
            *entity_sources: Variable number of entity lists
            
        Returns:
            Merged and deduplicated entity list
        """
        merged = []
        for source in entity_sources:
            merged.extend(source)
        
        if not merged:
            return []
        
        # Sort by priority (desc), then by length (desc)
        merged = sorted(
            merged,
            key=lambda x: (
                self.LABEL_PRIORITY.get(x.get("label", "MISC"), 0),
                len(x.get("text", ""))
            ),
            reverse=True
        )
        
        final = []
        seen_keys = set()
        
        for ent in merged:
            text = ent.get("text", "")
            label = ent.get("label", "MISC")
            
            if not text:
                continue
            
            # Normalize by removing whitespace for deduplication
            normalized_key = "".join(text.lower().split())
            
            if normalized_key in seen_keys:
                continue
            
            # Check if this entity is a substring of an already accepted entity
            is_redundant = False
            for accepted in final:
                accepted_text = accepted.get("text", "")
                accepted_key = "".join(accepted_text.lower().split())
                
                # If current entity text is contained in accepted entity, it's redundant
                if normalized_key in accepted_key:
                    is_redundant = True
                    break
            
            if is_redundant:
                continue
            
            # Add to final results
            final.append({
                "text": text,
                "label": label,
                "start": ent.get("start", 0)
            })
            seen_keys.add(normalized_key)
        
        # Sort by position in text (if available)
        final = sorted(final, key=lambda x: x.get("start", 0))
        
        return final
    
    def extract_with_metadata(self, text: str) -> Dict:
        """
        Extract entities and return with metadata.
        
        Returns:
            Dict with format:
            {
                "text": input_text,
                "entities": [...],
                "stats": {
                    "total_entities": int,
                    "by_label": {"PERSON": int, "ORG": int, ...}
                }
            }
        """
        entities = self.extract(text)
        
        # Compute statistics
        stats = {"total_entities": len(entities), "by_label": {}}
        for ent in entities:
            label = ent.get("label", "MISC")
            stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
        
        return {
            "text": text,
            "entities": entities,
            "stats": stats
        }
