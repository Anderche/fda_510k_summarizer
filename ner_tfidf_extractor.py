"""
NER and TF-IDF Extraction Module

Extracts Named Entities and TF-IDF features from queries for enhanced retrieval.
"""

import re
from typing import List, Dict, Any, Set
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
# Download POS tagger - try multiple resource names
_tagger_downloaded = False
for tagger_name in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'taggers/{tagger_name}')
        _tagger_downloaded = True
        break
    except LookupError:
        try:
            nltk.download(tagger_name, quiet=True)
            _tagger_downloaded = True
            break
        except:
            continue
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None


class NERTFIDFExtractor:
    """
    Extracts Named Entities and TF-IDF features from text.
    """
    
    def __init__(self):
        """Initialize NER and TF-IDF extractor."""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        self._corpus_fitted = False
    
    def extract_ner_nltk(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using NLTK (fallback method).
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'GPE': [],  # Geopolitical entity
            'MONEY': [],
            'DATE': [],
            'MEDICAL': []  # Medical terms
        }
        
        try:
            # Tokenize and tag
            tokens = nltk.word_tokenize(text)
            try:
                pos_tags = nltk.pos_tag(tokens)
            except LookupError:
                # Download missing tagger if needed (try multiple names)
                downloaded = False
                for tagger_name in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger']:
                    try:
                        nltk.download(tagger_name, quiet=True)
                        pos_tags = nltk.pos_tag(tokens)
                        downloaded = True
                        break
                    except:
                        continue
                if not downloaded:
                    # If still fails, create simple tags based on capitalization
                    pos_tags = [(token, 'NNP' if token and token[0].isupper() else 'NN') for token in tokens if token]
        except Exception as e:
            # If tokenization fails, use simple word splitting
            tokens = text.split()
            pos_tags = [(token, 'NNP' if token[0].isupper() else 'NN') for token in tokens]
        
        # Extract medical device terms and product codes
        medical_patterns = [
            r'\b[A-Z]{3}\b',  # Product codes like "NAY", "LZS"
            r'\bK\d+\b',  # K-numbers
            r'\b\d{3}\(k\)\b',  # 510(k)
            r'\b(?:device|implant|surgical|medical|robotic|software|hardware)\b',
            r'\b(?:FDA|CDRH|510k|510\(k\))\b'
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['MEDICAL'].extend(matches)
        
        # Simple noun phrase extraction for organizations
        for i, (word, pos) in enumerate(pos_tags):
            if pos in ['NNP', 'NNPS']:  # Proper nouns
                if len(word) > 2:
                    entities['ORGANIZATION'].append(word)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_ner_spacy(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using spaCy (preferred method if available).
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        if not SPACY_AVAILABLE or nlp is None:
            return self.extract_ner_nltk(text)
        
        doc = nlp(text)
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'GPE': [],
            'MONEY': [],
            'DATE': [],
            'MEDICAL': [],
            'PRODUCT': [],
            'LAW': []
        }
        
        # Extract standard entities
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # Extract medical device specific terms
        medical_patterns = [
            r'\b[A-Z]{3}\b',  # Product codes
            r'\bK\d+\b',  # K-numbers
            r'\b\d{3}\(k\)\b',  # 510(k)
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['MEDICAL'].extend(matches)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_ner(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (uses spaCy if available, else NLTK).
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with entity types and entities
        """
        if SPACY_AVAILABLE and nlp is not None:
            return self.extract_ner_spacy(text)
        else:
            return self.extract_ner_nltk(text)
    
    def extract_tfidf(self, text: str, corpus: List[str] = None) -> List[str]:
        """
        Extract top TF-IDF terms from text.
        
        Args:
            text: Text to extract TF-IDF terms from
            corpus: Optional corpus for TF-IDF calculation (if provided, fits on it)
            
        Returns:
            List of top TF-IDF terms/phrases
        """
        if corpus and not self._corpus_fitted:
            # Fit TF-IDF on corpus
            try:
                self.tfidf_vectorizer.fit(corpus)
                self._corpus_fitted = True
            except:
                pass
        
        # Transform query text
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = scores.argsort()[-10:][::-1]  # Top 10
            top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return top_terms
        except:
            # Fallback: extract important words manually
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            word_freq = Counter(words)
            top_terms = [word for word, _ in word_freq.most_common(10)]
            return top_terms
    
    def extract_query_features(self, query: str, corpus: List[str] = None) -> Dict[str, Any]:
        """
        Extract both NER and TF-IDF features from query.
        
        Args:
            query: Query text
            corpus: Optional corpus for TF-IDF
            
        Returns:
            Dictionary with 'ner' and 'tfidf' keys
        """
        ner_results = self.extract_ner(query)
        tfidf_terms = self.extract_tfidf(query, corpus)
        
        return {
            'ner': ner_results,
            'tfidf': tfidf_terms,
            'ner_text': self._format_ner(ner_results),
            'tfidf_text': ', '.join(tfidf_terms[:5]) if tfidf_terms else ''
        }
    
    def _format_ner(self, ner_dict: Dict[str, List[str]]) -> str:
        """Format NER results as readable text."""
        parts = []
        for entity_type, entities in ner_dict.items():
            if entities:
                parts.append(f"{entity_type}: {', '.join(entities[:5])}")
        return '; '.join(parts) if parts else 'None'

