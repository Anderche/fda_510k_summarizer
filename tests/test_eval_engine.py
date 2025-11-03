"""
Test script for the evaluation engine.
Demonstrates usage of ROUGE and BLEU metrics.
"""

import sys
from pathlib import Path

# Add project directories to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
config_path = project_root / "config"
sys.path.insert(0, str(config_path))
sys.path.insert(0, str(src_path))

from eval_engine import EvalEngine


def test_single_evaluation():
    """Test evaluation of a single candidate-reference pair."""
    print("=" * 80)
    print("Test 1: Single Evaluation")
    print("=" * 80)
    
    engine = EvalEngine()
    
    candidate = """
    The device is a medical pacemaker designed for cardiac rhythm management.
    It includes advanced safety features and has been tested in clinical trials.
    The FDA approved the device based on substantial equivalence to a predicate device.
    """
    
    reference = """
    This medical pacemaker device is used for cardiac rhythm management.
    The device has undergone extensive clinical testing and demonstrates safety.
    FDA approval was granted based on substantial equivalence demonstration.
    """
    
    results = engine.evaluate(candidate.strip(), reference.strip())
    print(engine.format_results(results))
    print()


def test_batch_evaluation():
    """Test batch evaluation of multiple pairs."""
    print("=" * 80)
    print("Test 2: Batch Evaluation")
    print("=" * 80)
    
    engine = EvalEngine()
    
    candidates = [
        "The device is a pacemaker with safety features.",
        "This medical device helps with heart rhythm problems.",
        "A cardiac pacemaker device for rhythm management."
    ]
    
    references = [
        "This is a pacemaker device designed for cardiac safety.",
        "The medical device assists with cardiac rhythm issues.",
        "Cardiac pacemaker device used for rhythm management."
    ]
    
    results = engine.evaluate_batch(candidates, references)
    print(engine.format_results(results))
    print()


def test_rouge_only():
    """Test ROUGE metrics only."""
    print("=" * 80)
    print("Test 3: ROUGE Metrics Only")
    print("=" * 80)
    
    engine = EvalEngine()
    
    candidate = "The pacemaker device has been approved by the FDA for cardiac use."
    reference = "FDA approved the cardiac pacemaker device for medical use."
    
    results = engine.evaluate(candidate, reference, metrics=['rouge'])
    print(engine.format_results(results))
    print()


def test_bleu_only():
    """Test BLEU metrics only."""
    print("=" * 80)
    print("Test 4: BLEU Metrics Only")
    print("=" * 80)
    
    engine = EvalEngine()
    
    candidate = "The pacemaker device has been approved by the FDA for cardiac use."
    reference = "FDA approved the cardiac pacemaker device for medical use."
    
    results = engine.evaluate(candidate, reference, metrics=['bleu'])
    print(engine.format_results(results))
    print()


if __name__ == "__main__":
    try:
        test_single_evaluation()
        test_batch_evaluation()
        test_rouge_only()
        test_bleu_only()
        print("=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

