"""
Build grouped token vocabulary from synth dataset.
This creates TOKEN_LIST with visual groups like "0^8" instead of separate tokens.
"""
import json
import sys
sys.path.insert(0, '.')

from backend.core.grouping import group_tokens, build_grouped_vocab

def main():
    print("Building grouped vocabulary from synth dataset...")

    # Load all expressions
    expressions = []

    for split in ["train", "val", "test"]:
        labels_path = f"data/synth/{split}/labels.jsonl"
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    expressions.append(data["target"])
        except FileNotFoundError:
            print(f"Warning: {labels_path} not found, skipping...")
            continue

    print(f"Loaded {len(expressions)} expressions")

    # Build vocabulary
    vocab = build_grouped_vocab(expressions)

    print(f"\nGrouped vocabulary size: {len(vocab)}")
    print(f"\nFirst 50 tokens:")
    for i, tok in enumerate(vocab[:50]):
        print(f"  {i:3d}: {tok}")

    # Show examples of grouped tokens
    print(f"\nExamples of grouped tokens (with ^):")
    grouped_examples = [tok for tok in vocab if "^" in tok][:20]
    for tok in grouped_examples:
        print(f"  {tok}")

    # Save to file
    with open("grouped_vocab.txt", "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(tok + "\n")

    print(f"\nVocabulary saved to grouped_vocab.txt")

    # Show before/after example
    print(f"\n" + "="*60)
    print("Example transformation:")
    example_expr = "0 ^ ( 8 ) + x ^ ( 2 )"
    example_tokens = example_expr.split()
    grouped = group_tokens(example_tokens)

    print(f"  Before: {example_expr}")
    print(f"          {example_tokens}")
    print(f"  After:  {' '.join(grouped)}")
    print(f"          {grouped}")
    print("="*60)

if __name__ == "__main__":
    main()
