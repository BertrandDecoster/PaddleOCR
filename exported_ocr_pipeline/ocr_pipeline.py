#!/usr/bin/env python3
"""
Standalone ONNX OCR Pipeline.

This module provides a simple API for OCR inference using ONNX models.
It uses PaddleOCR preprocessing and postprocessing for robust results.

Usage:
    from ocr_pipeline import OCRPipeline

    ocr = OCRPipeline("./models")
    results = ocr.predict("image.jpg")
    # Returns: [{"text": "...", "box": [[x,y],...], "confidence": 0.95}, ...]
"""

import math
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon


class DetPreprocess:
    """Detection model preprocessing."""

    def __init__(self, limit_side_len: int = 960, limit_type: str = "max"):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype("float32")
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype("float32")

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Preprocess image for detection.

        Args:
            img: Input image in BGR format (OpenCV default)

        Returns:
            Tuple of (preprocessed image, shape info dict)
        """
        src_h, src_w = img.shape[:2]

        # Resize
        img, (ratio_h, ratio_w) = self._resize(img)

        # Normalize
        img = img.astype("float32") / 255.0
        img = (img - self.mean) / self.std

        # HWC to CHW
        img = img.transpose((2, 0, 1))

        # Add batch dimension
        img = img[np.newaxis, :, :, :]

        shape_info = {
            "src_h": src_h,
            "src_w": src_w,
            "ratio_h": ratio_h,
            "ratio_w": ratio_w,
        }

        return img.astype("float32"), shape_info

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Resize image to be divisible by 32."""
        h, w = img.shape[:2]

        if self.limit_type == "max":
            if max(h, w) > self.limit_side_len:
                ratio = float(self.limit_side_len) / max(h, w)
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < self.limit_side_len:
                ratio = float(self.limit_side_len) / min(h, w)
            else:
                ratio = 1.0
        else:
            ratio = float(self.limit_side_len) / max(h, w)

        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)

        img = cv2.resize(img, (resize_w, resize_h))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, (ratio_h, ratio_w)


class DetPostprocess:
    """Detection model postprocessing (DB algorithm)."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        min_size: int = 3,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def __call__(
        self, pred: np.ndarray, shape_info: dict
    ) -> list[tuple[np.ndarray, float]]:
        """Postprocess detection output.

        Args:
            pred: Model output of shape (1, 1, H, W)
            shape_info: Dict with src_h, src_w, ratio_h, ratio_w

        Returns:
            List of (box, score) tuples where box is 4x2 array of corners
        """
        pred = pred[0, 0, :, :]
        segmentation = pred > self.thresh

        src_h = shape_info["src_h"]
        src_w = shape_info["src_w"]

        boxes, scores = self._boxes_from_bitmap(pred, segmentation, src_w, src_h)
        return list(zip(boxes, scores))

    def _boxes_from_bitmap(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> tuple[list[np.ndarray], list[float]]:
        """Extract boxes from bitmap."""
        height, width = bitmap.shape

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for i in range(num_contours):
            contour = contours[i]
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self._unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue

            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)

        return boxes, scores

    def _get_mini_boxes(self, contour: np.ndarray) -> tuple[list, float]:
        """Get minimum area bounding box."""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _box_score_fast(self, bitmap: np.ndarray, box: np.ndarray) -> float:
        """Calculate box score using mean of bitmap values in box region."""
        h, w = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _unclip(self, box: np.ndarray, unclip_ratio: float) -> list:
        """Expand box using Clipper library."""
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded


class RecPreprocess:
    """Recognition model preprocessing."""

    def __init__(self, image_shape: tuple[int, int, int] = (3, 48, 320)):
        self.image_shape = image_shape

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for recognition.

        Args:
            img: Input image in BGR format

        Returns:
            Preprocessed image of shape (1, C, H, W)
        """
        imgC, imgH, imgW = self.image_shape
        h, w = img.shape[:2]

        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")

        if imgC == 1:
            resized_image = resized_image / 255.0
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255.0

        # Normalize to [-1, 1]
        resized_image -= 0.5
        resized_image /= 0.5

        # Pad to target width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im[np.newaxis, :, :, :]


class RecPostprocess:
    """Recognition model postprocessing (CTC decode)."""

    def __init__(self, dict_path: str):
        self.characters = self._load_dict(dict_path)

    def _load_dict(self, dict_path: str) -> list[str]:
        """Load character dictionary."""
        characters = [""]  # Index 0 is blank token for CTC
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                char = line.strip("\n")
                if char:
                    characters.append(char)
        return characters

    def __call__(self, probs: np.ndarray) -> tuple[str, float]:
        """Decode CTC output to text.

        Args:
            probs: Model output of shape (1, T, num_classes)

        Returns:
            Tuple of (decoded text, average confidence)
        """
        probs = probs[0]  # Shape: (T, num_classes)

        # Greedy decode
        indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # CTC decode: remove blanks and consecutive duplicates
        decoded_chars = []
        decoded_confs = []
        prev_idx = -1

        for i, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:
                if idx < len(self.characters):
                    decoded_chars.append(self.characters[idx])
                    decoded_confs.append(confidences[i])
            prev_idx = idx

        text = "".join(decoded_chars)
        avg_conf = float(np.mean(decoded_confs)) if decoded_confs else 0.0

        return text, avg_conf


def get_perspective_transform(box: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Crop and perspective-correct a text region.

    Args:
        box: 4x2 array of corner points
        img: Source image

    Returns:
        Cropped and perspective-corrected image
    """
    box = box.astype(np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    # The box from detection is already ordered this way

    # Calculate output dimensions
    width = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3])
    ))
    height = int(max(
        np.linalg.norm(box[0] - box[3]),
        np.linalg.norm(box[1] - box[2])
    ))

    if width == 0 or height == 0:
        return None

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst)
    cropped = cv2.warpPerspective(img, M, (width, height))

    return cropped


class OCRPipeline:
    """ONNX-based OCR pipeline combining detection and recognition."""

    def __init__(
        self,
        model_dir: str,
        det_limit_side_len: int = 960,
        det_limit_type: str = "max",
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        det_unclip_ratio: float = 1.5,
    ):
        """Initialize the OCR pipeline.

        Args:
            model_dir: Path to directory containing models/ subdirectory
            det_limit_side_len: Max side length for detection resize
            det_limit_type: "max" or "min" for resize limit
            det_thresh: Detection binarization threshold
            det_box_thresh: Detection box score threshold
            det_unclip_ratio: Detection unclip ratio
        """
        model_path = Path(model_dir)
        models_path = model_path / "models"

        if not models_path.exists():
            # Try using model_dir directly
            models_path = model_path

        det_model_path = models_path / "det_model.onnx"
        rec_model_path = models_path / "rec_model.onnx"
        dict_path = models_path / "dictionary.txt"

        if not det_model_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_model_path}")
        if not rec_model_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {rec_model_path}")
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_path}")

        # Load ONNX models
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.det_session = ort.InferenceSession(str(det_model_path), providers=providers)
        self.rec_session = ort.InferenceSession(str(rec_model_path), providers=providers)

        # Get input/output names
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name

        # Initialize preprocessors and postprocessors
        self.det_preprocess = DetPreprocess(det_limit_side_len, det_limit_type)
        self.det_postprocess = DetPostprocess(
            det_thresh, det_box_thresh, 1000, det_unclip_ratio
        )
        self.rec_preprocess = RecPreprocess()
        self.rec_postprocess = RecPostprocess(str(dict_path))

        print(f"Loaded OCR pipeline from {model_dir}")
        print(f"  Detection model: {det_model_path}")
        print(f"  Recognition model: {rec_model_path}")
        print(f"  Using providers: {self.det_session.get_providers()}")

    def predict(
        self, image: Union[str, np.ndarray, list]
    ) -> list[dict]:
        """Run OCR on image(s).

        Args:
            image: Image path, numpy array (BGR), or list of either

        Returns:
            List of results, each with "text", "box", and "confidence" keys
        """
        # Handle single image
        if isinstance(image, (str, np.ndarray)):
            return self._predict_single(image)

        # Handle multiple images
        all_results = []
        for img in image:
            results = self._predict_single(img)
            all_results.append(results)
        return all_results

    def _predict_single(self, image: Union[str, np.ndarray]) -> list[dict]:
        """Run OCR on a single image."""
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image

        # Detection
        det_input, shape_info = self.det_preprocess(img)
        det_output = self.det_session.run(
            [self.det_output_name], {self.det_input_name: det_input}
        )[0]
        boxes_and_scores = self.det_postprocess(det_output, shape_info)

        if not boxes_and_scores:
            return []

        # Recognition for each box
        results = []
        for box, det_score in boxes_and_scores:
            # Crop and perspective correct
            cropped = get_perspective_transform(box, img)
            if cropped is None or cropped.size == 0:
                continue

            # Preprocess for recognition
            rec_input = self.rec_preprocess(cropped)

            # Run recognition
            rec_output = self.rec_session.run(
                [self.rec_output_name], {self.rec_input_name: rec_input}
            )[0]

            # Decode
            text, rec_conf = self.rec_postprocess(rec_output)

            if text:  # Only add non-empty results
                results.append({
                    "text": text,
                    "box": box.tolist(),
                    "confidence": float(rec_conf),
                })

        # Sort by vertical position (top to bottom)
        results.sort(key=lambda x: (min(p[1] for p in x["box"]), min(p[0] for p in x["box"])))

        return results


def main():
    """CLI entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ONNX OCR Pipeline")
    parser.add_argument("model_dir", help="Path to OCR pipeline directory")
    parser.add_argument("images", nargs="+", help="Image path(s) to process")
    parser.add_argument("-o", "--output", help="Output JSON file (optional)")
    parser.add_argument("--det-limit", type=int, default=960,
                        help="Detection max side length (default: 960)")
    parser.add_argument("--det-thresh", type=float, default=0.3,
                        help="Detection threshold (default: 0.3)")
    parser.add_argument("--det-box-thresh", type=float, default=0.6,
                        help="Detection box threshold (default: 0.6)")

    args = parser.parse_args()

    # Initialize pipeline
    ocr = OCRPipeline(
        args.model_dir,
        det_limit_side_len=args.det_limit,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
    )

    # Process images
    all_results = {}
    for image_path in args.images:
        print(f"\nProcessing: {image_path}")
        results = ocr.predict(image_path)
        all_results[image_path] = results

        for r in results:
            print(f"  {r['text']} (conf: {r['confidence']:.4f})")

    # Save output if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
