"""
Evaluation Engine Module

Provides ROUGE and BLEU metrics for evaluating generated summaries/responses
against reference texts.
"""

from typing import List, Dict, Any, Optional, Union
import csv
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score package not installed. Install with: pip install rouge-score")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class EvalEngine:
    """
    Evaluation engine for computing ROUGE and BLEU metrics.
    """
    
    def __init__(self):
        """Initialize the evaluation engine."""
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
            print("Warning: ROUGE metrics will not be available. Install rouge-score package.")
    
    def compute_rouge(self, candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores between candidate and reference texts.
        
        Args:
            candidate: Generated text to evaluate
            reference: Reference/ground truth text
            
        Returns:
            Dictionary with ROUGE scores (rouge1, rouge2, rougeL, rougeLsum)
            Each metric contains 'precision', 'recall', 'fmeasure'
        """
        if not ROUGE_AVAILABLE:
            raise ImportError(
                "rouge-score package is required for ROUGE metrics. "
                "Install with: pip install rouge-score"
            )
        
        if not candidate or not reference:
            return {
                'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeLsum': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
            }
        
        scores = self.rouge_scorer.score(reference, candidate)
        
        result = {}
        for metric_name, score in scores.items():
            result[metric_name] = {
                'precision': score.precision,
                'recall': score.recall,
                'fmeasure': score.fmeasure
            }
        
        return result
    
    def compute_bleu(self, candidate: str, reference: str, 
                   smoothing: bool = True) -> Dict[str, float]:
        """
        Compute BLEU score between candidate and reference texts.
        
        Args:
            candidate: Generated text to evaluate
            reference: Reference/ground truth text
            smoothing: Whether to use smoothing function (default: True)
            
        Returns:
            Dictionary with BLEU scores (bleu-1 through bleu-4, and overall bleu)
        """
        if not candidate or not reference:
            return {
                'bleu-1': 0.0,
                'bleu-2': 0.0,
                'bleu-3': 0.0,
                'bleu-4': 0.0,
                'bleu': 0.0
            }
        
        # Tokenize
        candidate_tokens = word_tokenize(candidate.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        if not candidate_tokens or not reference_tokens:
            return {
                'bleu-1': 0.0,
                'bleu-2': 0.0,
                'bleu-3': 0.0,
                'bleu-4': 0.0,
                'bleu': 0.0
            }
        
        smoothing_fn = SmoothingFunction().method1 if smoothing else None
        
        # Compute BLEU scores for n-grams 1-4
        scores = {}
        for n in range(1, 5):
            try:
                # Create n-gram references
                ref_ngrams = [reference_tokens[i:i+n] for i in range(len(reference_tokens) - n + 1)]
                cand_ngrams = [candidate_tokens[i:i+n] for i in range(len(candidate_tokens) - n + 1)]
                
                # Compute precision
                if cand_ngrams:
                    matches = sum(1 for ngram in cand_ngrams if ngram in ref_ngrams)
                    precision = matches / len(cand_ngrams)
                else:
                    precision = 0.0
                
                scores[f'bleu-{n}'] = precision
            except Exception:
                scores[f'bleu-{n}'] = 0.0
        
        # Compute overall BLEU score
        try:
            bleu_score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=smoothing_fn
            )
            scores['bleu'] = bleu_score
        except Exception:
            scores['bleu'] = 0.0
        
        return scores
    
    def evaluate(self, candidate: str, reference: str,
                 metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute all evaluation metrics (ROUGE and BLEU) for candidate vs reference.
        
        Args:
            candidate: Generated text to evaluate
            reference: Reference/ground truth text
            metrics: List of metrics to compute (default: ['rouge', 'bleu'])
                    Options: 'rouge', 'bleu', or both
            
        Returns:
            Dictionary containing all computed metrics
        """
        if metrics is None:
            metrics = ['rouge', 'bleu']
        
        result = {}
        
        if 'rouge' in metrics:
            try:
                result['rouge'] = self.compute_rouge(candidate, reference)
            except ImportError:
                result['rouge'] = {'error': 'ROUGE metrics not available (install rouge-score)'}
            except Exception as e:
                result['rouge'] = {'error': str(e)}
        
        if 'bleu' in metrics:
            try:
                result['bleu'] = self.compute_bleu(candidate, reference)
            except Exception as e:
                result['bleu'] = {'error': str(e)}
        
        return result
    
    def evaluate_batch(self, candidates: List[str], references: List[str],
                       metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate multiple candidate-reference pairs and compute aggregate statistics.
        
        Args:
            candidates: List of generated texts
            references: List of reference texts (must match candidates length)
            metrics: List of metrics to compute (default: ['rouge', 'bleu'])
            
        Returns:
            Dictionary with individual scores and aggregate statistics (mean, std)
        """
        if len(candidates) != len(references):
            raise ValueError(f"candidates and references must have same length. "
                           f"Got {len(candidates)} candidates and {len(references)} references")
        
        if metrics is None:
            metrics = ['rouge', 'bleu']
        
        all_scores = []
        
        for candidate, reference in zip(candidates, references):
            score = self.evaluate(candidate, reference, metrics=metrics)
            all_scores.append(score)
        
        # Compute aggregate statistics
        aggregate = {}
        
        if 'rouge' in metrics and ROUGE_AVAILABLE:
            rouge_aggregate = {}
            for metric_name in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                precisions = [s['rouge'][metric_name]['precision'] 
                             for s in all_scores if 'rouge' in s and metric_name in s['rouge'] 
                             and 'error' not in s['rouge']]
                recalls = [s['rouge'][metric_name]['recall'] 
                          for s in all_scores if 'rouge' in s and metric_name in s['rouge'] 
                          and 'error' not in s['rouge']]
                fmeasures = [s['rouge'][metric_name]['fmeasure'] 
                            for s in all_scores if 'rouge' in s and metric_name in s['rouge'] 
                            and 'error' not in s['rouge']]
                
                if precisions:
                    rouge_aggregate[metric_name] = {
                        'precision': {
                            'mean': sum(precisions) / len(precisions),
                            'std': self._std_dev(precisions),
                            'min': min(precisions),
                            'max': max(precisions)
                        },
                        'recall': {
                            'mean': sum(recalls) / len(recalls),
                            'std': self._std_dev(recalls),
                            'min': min(recalls),
                            'max': max(recalls)
                        },
                        'fmeasure': {
                            'mean': sum(fmeasures) / len(fmeasures),
                            'std': self._std_dev(fmeasures),
                            'min': min(fmeasures),
                            'max': max(fmeasures)
                        }
                    }
            aggregate['rouge'] = rouge_aggregate
        
        if 'bleu' in metrics:
            bleu_aggregate = {}
            for metric_name in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bleu']:
                scores = [s['bleu'][metric_name] 
                         for s in all_scores if 'bleu' in s and metric_name in s['bleu'] 
                         and 'error' not in s['bleu']]
                if scores:
                    bleu_aggregate[metric_name] = {
                        'mean': sum(scores) / len(scores),
                        'std': self._std_dev(scores),
                        'min': min(scores),
                        'max': max(scores)
                    }
            aggregate['bleu'] = bleu_aggregate
        
        return {
            'individual_scores': all_scores,
            'aggregate': aggregate,
            'num_samples': len(candidates)
        }
    
    def _std_dev(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def format_results(self, results: Dict[str, Any], verbose: bool = True) -> str:
        """
        Format evaluation results as a readable string.
        
        Args:
            results: Results dictionary from evaluate() or evaluate_batch()
            verbose: Whether to include detailed metrics (default: True)
            
        Returns:
            Formatted string with results
        """
        lines = []
        
        if 'individual_scores' in results:
            # Batch results
            lines.append(f"Evaluation Results (n={results['num_samples']})")
            lines.append("=" * 80)
            
            if 'aggregate' in results:
                agg = results['aggregate']
                
                if 'rouge' in agg:
                    lines.append("\nROUGE Scores (Aggregate):")
                    lines.append("-" * 80)
                    for metric_name, metrics in agg['rouge'].items():
                        lines.append(f"\n{metric_name.upper()}:")
                        for measure_name, stats in metrics.items():
                            lines.append(f"  {measure_name.capitalize()}: "
                                       f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                                       f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
                
                if 'bleu' in agg:
                    lines.append("\nBLEU Scores (Aggregate):")
                    lines.append("-" * 80)
                    for metric_name, stats in agg['bleu'].items():
                        lines.append(f"  {metric_name}: "
                                   f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                                   f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        else:
            # Single evaluation
            lines.append("Evaluation Results")
            lines.append("=" * 80)
            
            if 'rouge' in results:
                if 'error' in results['rouge']:
                    lines.append(f"\nROUGE: {results['rouge']['error']}")
                else:
                    lines.append("\nROUGE Scores:")
                    lines.append("-" * 80)
                    for metric_name, metrics in results['rouge'].items():
                        lines.append(f"\n{metric_name.upper()}:")
                        lines.append(f"  Precision: {metrics['precision']:.4f}")
                        lines.append(f"  Recall: {metrics['recall']:.4f}")
                        lines.append(f"  F-measure: {metrics['fmeasure']:.4f}")
            
            if 'bleu' in results:
                if 'error' in results['bleu']:
                    lines.append(f"\nBLEU: {results['bleu']['error']}")
                else:
                    lines.append("\nBLEU Scores:")
                    lines.append("-" * 80)
                    for metric_name, score in results['bleu'].items():
                        lines.append(f"  {metric_name}: {score:.4f}")
        
        return "\n".join(lines)
    
    def evaluate_from_csv(self, csv_path: str, query_pipeline, 
                          k: int = 5, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate RAG system by reading queries from CSV and comparing against expected responses.
        
        Args:
            csv_path: Path to CSV file with 'query' and 'expected_response' columns
            query_pipeline: QueryPipeline instance to generate candidates
            k: Number of chunks to retrieve for each query
            metrics: List of metrics to compute (default: ['rouge', 'bleu'])
            
        Returns:
            Dictionary with evaluation results (same format as evaluate_batch)
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        if metrics is None:
            metrics = ['rouge', 'bleu']
        
        queries = []
        references = []
        candidates = []
        
        # Read CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate headers
            if 'query' not in reader.fieldnames:
                raise ValueError("CSV must contain 'query' column")
            if 'expected_response' not in reader.fieldnames:
                raise ValueError("CSV must contain 'expected_response' column")
            
            # Process each row
            for row in reader:
                query = row['query'].strip()
                expected_response = row['expected_response'].strip()
                
                if not query or not expected_response:
                    continue  # Skip empty rows
                
                queries.append(query)
                references.append(expected_response)
                
                # Generate candidate response using RAG pipeline
                try:
                    result = query_pipeline.process_query(query, k=k)
                    candidate = result.get('response', '').strip()
                    candidates.append(candidate)
                except Exception as e:
                    print(f"Warning: Failed to process query '{query}': {e}")
                    candidates.append("")  # Empty candidate on error
        
        # Evaluate batch
        batch_results = self.evaluate_batch(candidates, references, metrics=metrics)
        
        # Add query information to results
        batch_results['queries'] = queries
        batch_results['candidates'] = candidates
        
        return batch_results

