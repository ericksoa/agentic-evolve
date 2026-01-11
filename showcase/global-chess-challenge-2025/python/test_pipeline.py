"""
Quick Pipeline Test

Tests that all components work without running full training.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")

    try:
        import chess
        import chess.pgn
        import chess.engine
        print("  chess: OK")
    except ImportError as e:
        print(f"  chess: FAIL - {e}")
        return False

    try:
        import torch
        print(f"  torch: OK (version {torch.__version__})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        print(f"    MPS available: {torch.backends.mps.is_available()}")
    except ImportError as e:
        print(f"  torch: FAIL - {e}")
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  transformers: OK")
    except ImportError as e:
        print(f"  transformers: FAIL - {e}")
        return False

    try:
        from peft import LoraConfig, get_peft_model
        print("  peft: OK")
    except ImportError as e:
        print(f"  peft: FAIL - {e}")
        return False

    try:
        from datasets import Dataset
        print("  datasets: OK")
    except ImportError as e:
        print(f"  datasets: FAIL - {e}")
        return False

    try:
        import bitsandbytes
        print("  bitsandbytes: OK")
    except ImportError as e:
        print(f"  bitsandbytes: FAIL - {e}")
        return False

    return True


def test_stockfish():
    """Test Stockfish connection"""
    print("\nTesting Stockfish...")

    import chess
    import chess.engine

    try:
        engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
        board = chess.Board()
        result = engine.analyse(board, chess.engine.Limit(depth=10))
        print(f"  Stockfish: OK")
        print(f"    Best move: {result['pv'][0]}")
        engine.quit()
        return True
    except Exception as e:
        print(f"  Stockfish: FAIL - {e}")
        return False


def test_tokenizer():
    """Test tokenizer loading"""
    print("\nTesting tokenizer...")

    try:
        from transformers import AutoTokenizer

        # Use a smaller model for testing
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokens = tokenizer("Test chess prompt", return_tensors="pt")
        print(f"  Tokenizer: OK")
        print(f"    Vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        print(f"  Tokenizer: FAIL - {e}")
        return False


def test_training_data():
    """Test training data loading"""
    print("\nTesting training data...")

    data_path = Path(__file__).parent.parent / "data" / "training_test.jsonl"

    if not data_path.exists():
        print(f"  Training data: SKIP - {data_path} not found")
        print("  Run: python extract_training_data.py --pgn ... to create test data")
        return True  # Not a failure

    import json
    try:
        with open(data_path) as f:
            examples = [json.loads(line) for line in f]
        print(f"  Training data: OK")
        print(f"    Examples: {len(examples)}")
        return True
    except Exception as e:
        print(f"  Training data: FAIL - {e}")
        return False


def main():
    print("=" * 60)
    print("CHESS TRAINING PIPELINE TEST")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_stockfish()
    all_passed &= test_tokenizer()
    all_passed &= test_training_data()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Extract training data:")
        print("   python extract_training_data.py --pgn data/games/*.pgn --output data/training.jsonl --max-examples 5000")
        print("\n2. For local training (slow, needs GPU):")
        print("   python train_model.py --data data/training.jsonl --epochs 1")
        print("\n3. For cloud training (recommended):")
        print("   - Upload training data to Lambda Labs / RunPod")
        print("   - Run training on cloud GPU")
    else:
        print("SOME TESTS FAILED")
        print("=" * 60)
        print("\nFix the issues above before proceeding.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
