"""
Character segmentation for classical OCR approach.
Uses connected components analysis to extract individual characters.
"""
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image


def segment_characters(image: np.ndarray, min_area: int = 20, max_area: int = 5000) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment characters from a grayscale image using projection profiles + connected components.

    Strategy:
    1. Use vertical projection to find character gaps (works well for horizontal text)
    2. Within each vertical slice, use connected components for multi-character detection
    3. Fallback to pure connected components if projection fails

    Args:
        image: Grayscale numpy array (H, W), values 0-255
        min_area: Minimum component area in pixels
        max_area: Maximum component area in pixels

    Returns:
        List of (cropped_char_image, bbox) where bbox is (x, y, w, h)
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Binarize using Otsu's method
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert if background is white (we want black background, white foreground for ink detection)
    if binary[0, 0] > 127:  # top-left corner is likely background
        binary = 255 - binary

    # Apply morphological opening to separate touching characters slightly
    kernel_small = np.ones((2, 1), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # Find connected components with more aggressive separation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cleaned, connectivity=4)

    segments = []

    # Skip label 0 (background)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        # Very permissive area filter - accept almost anything
        if area < 5 or area > max_area:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Extract character region from original image (not binary)
        char_region = image[y:y+h, x:x+w].copy()

        segments.append((char_region, (x, y, w, h)))

    # Sort by x-coordinate (left to right)
    segments.sort(key=lambda item: item[1][0])

    return segments


def prepare_segment_for_classification(
    segment: np.ndarray,
    target_size: Tuple[int, int] = (32, 32),
    padding: int = 4
) -> np.ndarray:
    """
    Prepare a segmented character image for MLP classification.

    Args:
        segment: Grayscale character image (H, W)
        target_size: Target size for classifier (height, width)
        padding: Padding around character in pixels

    Returns:
        Normalized float32 array of shape (target_h, target_w) in range [0, 1]
    """
    h, w = segment.shape
    target_h, target_w = target_size

    # Add padding
    padded = np.pad(segment, padding, mode='constant', constant_values=255)

    # Calculate scale to fit in target size while maintaining aspect ratio
    scale = min((target_h - 2*padding) / padded.shape[0], (target_w - 2*padding) / padded.shape[1])
    new_h = max(1, int(padded.shape[0] * scale))
    new_w = max(1, int(padded.shape[1] * scale))

    # Resize
    pil_img = Image.fromarray(padded)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    resized_arr = np.array(resized)

    # Create canvas and center the character
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_arr

    # Normalize to [0, 1], invert so character is white on black background
    normalized = (255 - canvas).astype(np.float32) / 255.0

    return normalized


def segment_and_prepare(
    image: np.ndarray,
    target_size: Tuple[int, int] = (32, 32),
    min_area: int = 20,
    max_area: int = 5000
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment characters and prepare each for classification in one go.

    Returns:
        List of (prepared_image, bbox) ready for MLP classification
    """
    segments = segment_characters(image, min_area=min_area, max_area=max_area)

    prepared = []
    for char_img, bbox in segments:
        prep_img = prepare_segment_for_classification(char_img, target_size=target_size)
        prepared.append((prep_img, bbox))

    return prepared
