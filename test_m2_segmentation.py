"""
Test script for M2 segmentation functionality.
Verifies that segmentation works correctly on sample data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from PIL import Image
from backend.core.segmentation import segment_characters, prepare_segment_for_classification, segment_and_prepare

def test_segmentation():
    """Test segmentation on first 5 training samples."""
    print("=" * 60)
    print("Testing M2 Segmentation")
    print("=" * 60)

    labels_path = "data/synth/train/labels.jsonl"
    if not os.path.exists(labels_path):
        print(f"ERROR: {labels_path} not found!")
        return False

    # Load first 5 samples
    with open(labels_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f.readlines()[:5]]

    print(f"\nTesting {len(samples)} samples...\n")

    success_count = 0
    fail_count = 0

    for i, sample in enumerate(samples):
        img_path = os.path.join("data/synth/train", sample["image"])
        target_tokens = sample["target"].split()

        print(f"Sample {i+1}:")
        print(f"  Image: {sample['image']}")
        print(f"  Target: {sample['target']}")
        print(f"  Tokens: {len(target_tokens)}")

        if not os.path.exists(img_path):
            print(f"  ERROR: Image not found: {img_path}")
            fail_count += 1
            continue

        # Load and segment
        img = Image.open(img_path).convert("L")
        img_arr = np.array(img)

        segments = segment_characters(img_arr, min_area=15, max_area=4000)

        print(f"  Segments found: {len(segments)}")

        if len(segments) == 0:
            print(f"  WARNING: No segments found!")
            fail_count += 1
            continue

        if len(segments) == len(target_tokens):
            print(f"  OK: Segment count matches token count")
            success_count += 1
        else:
            print(f"  WARNING: Mismatch: {len(segments)} segments != {len(target_tokens)} tokens")
            fail_count += 1

        # Show segment details
        for j, (char_img, bbox) in enumerate(segments[:5]):  # First 5 segments
            print(f"    Seg {j}: bbox={bbox}, shape={char_img.shape}")

        print()

    print("=" * 60)
    print(f"Results: {success_count} OK, {fail_count} Failed")
    print("=" * 60)

    return success_count > 0


def test_dataset_loading():
    """Test that SegmentedCharDataset loads correctly."""
    print("\n" + "=" * 60)
    print("Testing SegmentedCharDataset")
    print("=" * 60)

    try:
        from backend.core.data.seg_dataset import SegmentedCharDataset

        print("\nCreating dataset with max_samples=10...")
        dataset = SegmentedCharDataset(
            data_dir="data/synth/train",
            target_size=(32, 32),
            max_samples=10
        )

        print(f"\nOK: Dataset created successfully!")
        print(f"  Total characters extracted: {len(dataset)}")

        if len(dataset) > 0:
            # Test first sample
            img, label = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Image shape: {img.shape}")
            print(f"  Label: {label}")
            print(f"  Image dtype: {img.dtype}")
            print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
            return True
        else:
            print("ERROR: No characters extracted!")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nM2 Segmentation Test Suite\n")

    test1 = test_segmentation()
    test2 = test_dataset_loading()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("OK: ALL TESTS PASSED")
        print("M2 segmentation is ready for training!")
    else:
        print("FAILED: SOME TESTS FAILED")
        print("Review errors above before training.")
    print("=" * 60)
