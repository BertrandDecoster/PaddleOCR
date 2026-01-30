#!/usr/bin/env python3
"""
Export PaddleOCR pipeline as standalone ONNX package.

This script creates a self-contained OCR pipeline folder with:
- ONNX models for detection and recognition
- Python inference wrapper using PaddleOCR preprocessing/postprocessing
- CLI entry point
- Requirements and documentation

Usage:
    python tools/export_onnx/export_pipeline.py \
        --rec-model ./output/server_rec_quick/model/best_accuracy \
        --rec-config ./output/server_rec_quick/model/config.yml \
        --output ./exported_ocr_pipeline

    # With custom detection model
    python tools/export_onnx/export_pipeline.py \
        --rec-model ./output/my_model/model/best_accuracy \
        --rec-config ./output/my_model/model/config.yml \
        --det-model ./my_det_model \
        --det-config ./configs/det/PP-OCRv5/PP-OCRv5_server_det.yml \
        --output ./exported_ocr_pipeline
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Add parent directories to path
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))


# URLs for pretrained detection models
DET_MODEL_URLS = {
    "PP-OCRv5_server_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar",
    "PP-OCRv5_mobile_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar",
}


def download_and_extract_model(url: str, dest_dir: str) -> str:
    """Download and extract a model tar file.

    Args:
        url: URL to download from
        dest_dir: Directory to extract to

    Returns:
        Path to extracted model directory
    """
    print(f"Downloading model from {url}...")
    os.makedirs(dest_dir, exist_ok=True)

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        tmp_path = tmp.name

    try:
        # Extract
        print(f"Extracting to {dest_dir}...")
        with tarfile.open(tmp_path, "r") as tar:
            tar.extractall(dest_dir)

        # Find extracted directory
        for item in os.listdir(dest_dir):
            item_path = os.path.join(dest_dir, item)
            if os.path.isdir(item_path):
                return item_path

        raise RuntimeError(f"No directory found after extraction in {dest_dir}")
    finally:
        os.unlink(tmp_path)


def export_paddle_to_onnx(
    model_dir: str,
    output_path: str,
    opset_version: int = 14,
) -> None:
    """Convert Paddle inference model to ONNX.

    Args:
        model_dir: Path to Paddle inference model directory
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    import paddle2onnx

    # Find model files
    model_file = None
    params_file = None

    for f in os.listdir(model_dir):
        if f.endswith(".pdmodel"):
            model_file = os.path.join(model_dir, f)
        elif f.endswith(".json") and "inference" in f:
            model_file = os.path.join(model_dir, f)
        elif f.endswith(".pdiparams"):
            params_file = os.path.join(model_dir, f)

    if model_file is None or params_file is None:
        raise ValueError(f"Could not find model files in {model_dir}")

    print(f"Converting {model_dir} to ONNX...")
    print(f"  Model file: {model_file}")
    print(f"  Params file: {params_file}")

    # Use paddle2onnx Python API
    paddle2onnx.export(
        model_filename=model_file,
        params_filename=params_file,
        save_file=output_path,
        opset_version=opset_version,
        enable_onnx_checker=True,
    )

    print(f"  Saved ONNX model to {output_path}")


def export_recognition_model(
    model_path: str,
    config_path: str,
    output_dir: str,
) -> str:
    """Export recognition model to Paddle inference format.

    Args:
        model_path: Path to trained model weights (without extension)
        config_path: Path to config YAML file
        output_dir: Directory to save inference model

    Returns:
        Path to inference model directory
    """
    print(f"Exporting recognition model from {model_path}...")

    inference_dir = os.path.join(output_dir, "rec_inference")
    os.makedirs(inference_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        config_path,
        "-o",
        f"Global.checkpoints={model_path}",
        "-o",
        f"Global.save_inference_dir={inference_dir}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(__dir__, "../.."))
    if result.returncode != 0:
        print(f"export_model.py stdout: {result.stdout}")
        print(f"export_model.py stderr: {result.stderr}")
        raise RuntimeError(f"export_model.py failed with code {result.returncode}")

    print(f"  Saved Paddle inference model to {inference_dir}")
    return inference_dir


def get_ocr_pipeline_template() -> str:
    """Return the ocr_pipeline.py template code."""
    return '''#!/usr/bin/env python3
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
                char = line.strip("\\n")
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
        print(f"\\nProcessing: {image_path}")
        results = ocr.predict(image_path)
        all_results[image_path] = results

        for r in results:
            print(f"  {r['text']} (conf: {r['confidence']:.4f})")

    # Save output if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
'''


def get_main_template() -> str:
    """Return the __main__.py template code."""
    return '''#!/usr/bin/env python3
"""CLI entry point for OCR pipeline."""

from ocr_pipeline import main

if __name__ == "__main__":
    main()
'''


def get_requirements_template() -> str:
    """Return the requirements.txt template."""
    return '''# ONNX OCR Pipeline Dependencies
onnxruntime>=1.15.0
# For GPU support, use: onnxruntime-gpu>=1.15.0
opencv-python>=4.5.0
numpy>=1.20.0
shapely>=2.0.0
pyclipper>=1.3.0
'''


def get_readme_template() -> str:
    """Return the README.md template."""
    return '''# ONNX OCR Pipeline

A standalone OCR pipeline using ONNX models for text detection and recognition.

## Installation

```bash
pip install -r requirements.txt

# For GPU support:
pip install onnxruntime-gpu
```

## Usage

### Python API

```python
from ocr_pipeline import OCRPipeline

# Initialize pipeline
ocr = OCRPipeline(".")

# Process single image
results = ocr.predict("image.jpg")
for r in results:
    print(f"Text: {r['text']}, Confidence: {r['confidence']:.2f}")
    print(f"Box: {r['box']}")

# Process multiple images
results = ocr.predict(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### CLI

```bash
# Process single image
python -m ocr_pipeline . image.jpg

# Process multiple images
python -m ocr_pipeline . img1.jpg img2.jpg img3.jpg

# Save results to JSON
python -m ocr_pipeline . image.jpg -o results.json

# Adjust detection parameters
python -m ocr_pipeline . image.jpg --det-limit 1280 --det-thresh 0.2
```

## Output Format

Results are returned as a list of dictionaries:

```python
[
    {
        "text": "detected text",
        "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # Quadrilateral corners
        "confidence": 0.95
    },
    ...
]
```

## Model Files

```
models/
├── det_model.onnx      # Text detection model (DB algorithm)
├── rec_model.onnx      # Text recognition model (CTC-based)
└── dictionary.txt      # Character dictionary
```

## Configuration

The `OCRPipeline` class accepts the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_dir` | (required) | Path to pipeline directory |
| `det_limit_side_len` | 960 | Max side length for detection resize |
| `det_limit_type` | "max" | Resize limit type: "max" or "min" |
| `det_thresh` | 0.3 | Detection binarization threshold |
| `det_box_thresh` | 0.6 | Detection box score threshold |
| `det_unclip_ratio` | 1.5 | Box expansion ratio |

## Generated By

This pipeline was exported from PaddleOCR using:
```bash
python tools/export_onnx/export_pipeline.py
```
'''


def main():
    parser = argparse.ArgumentParser(
        description="Export PaddleOCR pipeline as standalone ONNX package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rec-model",
        required=True,
        help="Path to recognition model weights (without extension)",
    )
    parser.add_argument(
        "--rec-config",
        required=True,
        help="Path to recognition model config YAML",
    )
    parser.add_argument(
        "--det-model",
        default=None,
        help="Path to detection model (Paddle inference format). If not provided, downloads PP-OCRv5_server_det",
    )
    parser.add_argument(
        "--det-config",
        default=None,
        help="Path to detection config YAML (only needed if --det-model points to weights)",
    )
    parser.add_argument(
        "--det-name",
        default="PP-OCRv5_server_det",
        choices=list(DET_MODEL_URLS.keys()),
        help="Pretrained detection model to use (default: PP-OCRv5_server_det)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for the exported pipeline",
    )
    parser.add_argument(
        "--dict",
        default=None,
        help="Path to character dictionary (default: ppocr/utils/dict/ppocrv5_dict.txt)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--test",
        nargs="+",
        metavar="IMAGE",
        help="Test image path(s) to verify ONNX output matches Paddle inference",
    )

    args = parser.parse_args()

    # Create output directory structure
    output_dir = Path(args.output)
    models_dir = output_dir / "models"
    os.makedirs(models_dir, exist_ok=True)

    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="ocr_export_")
    print(f"Using temp directory: {temp_dir}")

    try:
        # Get detection model
        if args.det_model:
            det_inference_dir = args.det_model
            if not os.path.isdir(det_inference_dir):
                raise ValueError(f"Detection model directory not found: {det_inference_dir}")
        else:
            # Download pretrained model
            det_url = DET_MODEL_URLS[args.det_name]
            det_inference_dir = download_and_extract_model(det_url, temp_dir)

        # Export recognition model to Paddle inference format
        rec_inference_dir = export_recognition_model(
            args.rec_model,
            args.rec_config,
            temp_dir,
        )

        # Convert detection model to ONNX
        det_onnx_path = str(models_dir / "det_model.onnx")
        export_paddle_to_onnx(det_inference_dir, det_onnx_path, args.opset)

        # Convert recognition model to ONNX
        rec_onnx_path = str(models_dir / "rec_model.onnx")
        export_paddle_to_onnx(rec_inference_dir, rec_onnx_path, args.opset)

        # Copy dictionary file
        if args.dict:
            dict_src = args.dict
        else:
            dict_src = os.path.join(__dir__, "../../ppocr/utils/dict/ppocrv5_dict.txt")

        dict_dst = models_dir / "dictionary.txt"
        shutil.copy(dict_src, dict_dst)
        print(f"Copied dictionary to {dict_dst}")

        # Write template files
        with open(output_dir / "ocr_pipeline.py", "w") as f:
            f.write(get_ocr_pipeline_template())
        print(f"Created {output_dir / 'ocr_pipeline.py'}")

        with open(output_dir / "__main__.py", "w") as f:
            f.write(get_main_template())
        print(f"Created {output_dir / '__main__.py'}")

        with open(output_dir / "requirements.txt", "w") as f:
            f.write(get_requirements_template())
        print(f"Created {output_dir / 'requirements.txt'}")

        with open(output_dir / "README.md", "w") as f:
            f.write(get_readme_template())
        print(f"Created {output_dir / 'README.md'}")

        # Print summary
        print("\n" + "=" * 60)
        print("Export complete!")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")
        print("\nContents:")
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                rel_path = item.relative_to(output_dir)
                print(f"  {rel_path} ({size_str})")

        print("\nUsage:")
        print(f"  cd {output_dir}")
        print("  pip install -r requirements.txt")
        print(f"  python -m ocr_pipeline . <image.jpg>")
        print("\nOr in Python:")
        print(f'  from {output_dir.name}.ocr_pipeline import OCRPipeline')
        print(f'  ocr = OCRPipeline("{output_dir}")')
        print('  results = ocr.predict("image.jpg")')

        # Run verification test if requested (before cleanup)
        test_results = None
        if args.test:
            test_results = run_verification_test(
                str(output_dir),
                args.test,
                det_inference_dir,
                rec_inference_dir,
                str(dict_dst),
            )

    except Exception as e:
        # Cleanup temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\nCleaned up temp directory: {temp_dir}")


def run_verification_test(
    output_dir: str,
    test_images: list[str],
    det_inference_dir: str,
    rec_inference_dir: str,
    dict_path: str,
) -> dict:
    """Run verification test to ensure ONNX pipeline produces valid results.

    This test:
    1. Runs the ONNX pipeline on test images
    2. Verifies that results are produced
    3. Checks that detected text is non-empty

    Args:
        output_dir: Path to exported pipeline directory
        test_images: List of test image paths
        det_inference_dir: Path to Paddle detection inference model (unused, for API compat)
        rec_inference_dir: Path to Paddle recognition inference model (unused, for API compat)
        dict_path: Path to character dictionary

    Returns:
        Dict with test results
    """
    import cv2
    import numpy as np
    import onnxruntime as ort

    # Import the exported pipeline
    sys.path.insert(0, str(Path(output_dir).parent))
    pipeline_module = __import__(Path(output_dir).name + ".ocr_pipeline", fromlist=["OCRPipeline"])
    OCRPipeline = pipeline_module.OCRPipeline

    print("\n" + "=" * 60)
    print("Running verification test")
    print("=" * 60)

    # Initialize ONNX pipeline
    print("\nLoading ONNX pipeline...")
    onnx_pipeline = OCRPipeline(output_dir)

    # Also load ONNX models directly to compare raw outputs
    print("Loading ONNX models for raw output comparison...")
    models_dir = Path(output_dir) / "models"
    det_session = ort.InferenceSession(
        str(models_dir / "det_model.onnx"),
        providers=["CPUExecutionProvider"]
    )
    rec_session = ort.InferenceSession(
        str(models_dir / "rec_model.onnx"),
        providers=["CPUExecutionProvider"]
    )

    # Load Paddle inference ONNX for comparison (convert on the fly if not exists)
    print("Converting Paddle models to ONNX for comparison...")
    import paddle2onnx

    # Find and convert detection model
    det_model_file = None
    det_params_file = None
    for f in os.listdir(det_inference_dir):
        if f.endswith(".pdmodel") or (f.endswith(".json") and "inference" in f):
            det_model_file = os.path.join(det_inference_dir, f)
        elif f.endswith(".pdiparams"):
            det_params_file = os.path.join(det_inference_dir, f)

    paddle_det_onnx = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    paddle2onnx.export(
        model_filename=det_model_file,
        params_filename=det_params_file,
        save_file=paddle_det_onnx.name,
        opset_version=14,
        enable_onnx_checker=True,
    )
    paddle_det_session = ort.InferenceSession(
        paddle_det_onnx.name,
        providers=["CPUExecutionProvider"]
    )

    # Find and convert recognition model
    rec_model_file = None
    rec_params_file = None
    for f in os.listdir(rec_inference_dir):
        if f.endswith(".pdmodel") or (f.endswith(".json") and "inference" in f):
            rec_model_file = os.path.join(rec_inference_dir, f)
        elif f.endswith(".pdiparams"):
            rec_params_file = os.path.join(rec_inference_dir, f)

    paddle_rec_onnx = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    paddle2onnx.export(
        model_filename=rec_model_file,
        params_filename=rec_params_file,
        save_file=paddle_rec_onnx.name,
        opset_version=14,
        enable_onnx_checker=True,
    )
    paddle_rec_session = ort.InferenceSession(
        paddle_rec_onnx.name,
        providers=["CPUExecutionProvider"]
    )

    # Detection preprocessing
    def det_preprocess(img, limit_side_len=960):
        src_h, src_w = img.shape[:2]
        ratio = float(limit_side_len) / max(src_h, src_w) if max(src_h, src_w) > limit_side_len else 1.0
        resize_h = max(int(round(src_h * ratio / 32) * 32), 32)
        resize_w = max(int(round(src_w * ratio / 32) * 32), 32)
        img_resized = cv2.resize(img, (resize_w, resize_h))

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype("float32")
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype("float32")
        img_norm = (img_resized.astype("float32") / 255.0 - mean) / std
        img_chw = img_norm.transpose((2, 0, 1))[np.newaxis, :, :, :].astype("float32")

        return img_chw, {"src_h": src_h, "src_w": src_w, "ratio_h": resize_h / src_h, "ratio_w": resize_w / src_w}

    # Recognition preprocessing
    def rec_preprocess(img, image_shape=(3, 48, 320)):
        import math
        imgC, imgH, imgW = image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(imgW, int(math.ceil(imgH * ratio)))
        resized_image = cv2.resize(img, (resized_w, imgH)).astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im[np.newaxis, :, :, :]

    results = {
        "total_images": len(test_images),
        "total_detections": 0,
        "det_outputs_match": 0,
        "rec_outputs_match": 0,
        "texts_found": [],
        "details": [],
    }

    for image_path in test_images:
        print(f"\nTesting: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ERROR: Could not read image")
            continue

        # Run ONNX pipeline
        pipeline_results = onnx_pipeline.predict(image_path)
        results["total_detections"] += len(pipeline_results)

        for r in pipeline_results:
            results["texts_found"].append(r["text"])
            print(f"  Detected: '{r['text']}' (conf: {r['confidence']:.4f})")

        # Compare detection model outputs
        det_input, shape_info = det_preprocess(img)

        det_input_name = det_session.get_inputs()[0].name
        det_output_name = det_session.get_outputs()[0].name
        exported_det_out = det_session.run([det_output_name], {det_input_name: det_input})[0]

        paddle_det_input_name = paddle_det_session.get_inputs()[0].name
        paddle_det_output_name = paddle_det_session.get_outputs()[0].name
        paddle_det_out = paddle_det_session.run([paddle_det_output_name], {paddle_det_input_name: det_input})[0]

        det_match = np.allclose(exported_det_out, paddle_det_out, rtol=1e-4, atol=1e-5)
        det_max_diff = np.max(np.abs(exported_det_out - paddle_det_out))
        if det_match:
            results["det_outputs_match"] += 1
        print(f"  Detection outputs match: {det_match} (max diff: {det_max_diff:.2e})")

        # Compare recognition model outputs on a cropped region (if any detected)
        if pipeline_results:
            # Import utilities from exported pipeline
            from exported_ocr_pipeline.ocr_pipeline import get_perspective_transform

            box = np.array(pipeline_results[0]["box"])
            cropped = get_perspective_transform(box, img)
            if cropped is not None and cropped.size > 0:
                rec_input = rec_preprocess(cropped)

                rec_input_name = rec_session.get_inputs()[0].name
                rec_output_name = rec_session.get_outputs()[0].name
                exported_rec_out = rec_session.run([rec_output_name], {rec_input_name: rec_input})[0]

                paddle_rec_input_name = paddle_rec_session.get_inputs()[0].name
                paddle_rec_output_name = paddle_rec_session.get_outputs()[0].name
                paddle_rec_out = paddle_rec_session.run([paddle_rec_output_name], {paddle_rec_input_name: rec_input})[0]

                rec_match = np.allclose(exported_rec_out, paddle_rec_out, rtol=1e-4, atol=1e-5)
                rec_max_diff = np.max(np.abs(exported_rec_out - paddle_rec_out))
                if rec_match:
                    results["rec_outputs_match"] += 1
                print(f"  Recognition outputs match: {rec_match} (max diff: {rec_max_diff:.2e})")

        results["details"].append({
            "image": image_path,
            "results": pipeline_results,
        })

    # Cleanup temp ONNX files
    os.unlink(paddle_det_onnx.name)
    os.unlink(paddle_rec_onnx.name)

    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"Total images tested: {results['total_images']}")
    print(f"Total text regions detected: {results['total_detections']}")
    print(f"Detection outputs match Paddle: {results['det_outputs_match']}/{results['total_images']}")
    print(f"Recognition outputs match Paddle: {results['rec_outputs_match']}/{results['total_images']}")

    if results["texts_found"]:
        print(f"\nTexts found: {results['texts_found']}")

    success = (
        results["total_detections"] > 0 and
        results["det_outputs_match"] == results["total_images"] and
        results["rec_outputs_match"] == results["total_images"]
    )
    print(f"\nVerification: {'PASSED' if success else 'FAILED'}")

    return results


def test_main():
    """Run test with exported pipeline."""
    parser = argparse.ArgumentParser(
        description="Test exported ONNX OCR pipeline against Paddle inference",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Path to exported pipeline directory",
    )
    parser.add_argument(
        "--det-inference",
        required=True,
        help="Path to Paddle detection inference model directory",
    )
    parser.add_argument(
        "--rec-inference",
        required=True,
        help="Path to Paddle recognition inference model directory",
    )
    parser.add_argument(
        "--dict",
        required=True,
        help="Path to character dictionary",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Test image paths",
    )

    args = parser.parse_args()

    results = run_verification_test(
        args.pipeline,
        args.images,
        args.det_inference,
        args.rec_inference,
        args.dict,
    )

    # Exit with error code if test failed
    success = results["text_matches"] > 0 and len(results["text_mismatches"]) == 0
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.argv.pop(1)  # Remove --test flag
        test_main()
    else:
        main()
