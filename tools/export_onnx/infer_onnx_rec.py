#!/usr/bin/env python3
"""
ONNX inference script for PP-OCRv5 recognition model.

This script performs text recognition inference using an exported ONNX model.
It includes the same preprocessing as PaddleOCR and CTC decoding for postprocessing.

Usage:
    python tools/export_onnx/infer_onnx_rec.py \
        --model output/license_plate_big/onnx/model.onnx \
        --dict output/license_plate_big/onnx/ppocrv5_dict.txt \
        --image path/to/image.jpg

    # Process multiple images
    python tools/export_onnx/infer_onnx_rec.py \
        --model output/license_plate_big/onnx/model.onnx \
        --dict output/license_plate_big/onnx/ppocrv5_dict.txt \
        --image img1.jpg img2.jpg img3.jpg

    # Verify ONNX export matches Paddle inference
    python tools/export_onnx/infer_onnx_rec.py \
        --model output/license_plate_big/onnx/model.onnx \
        --dict output/license_plate_big/onnx/ppocrv5_dict.txt \
        --verify path/to/image_or_folder \
        --paddle-model output/license_plate_big/inference_old
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_character_dict(dict_path: str) -> list[str]:
    """Load character dictionary from file.

    The dictionary file contains one character per line.
    Index 0 is reserved for the CTC blank token.

    Args:
        dict_path: Path to character dictionary file

    Returns:
        List of characters where index 0 is blank token
    """
    characters = [""]  # Index 0 is blank token for CTC
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            char = line.strip("\n")
            if char:
                characters.append(char)
    return characters


def resize_norm_img(img: np.ndarray, image_shape: tuple[int, int, int],
                    padding: bool = True) -> tuple[np.ndarray, float]:
    """Resize and normalize image for recognition.

    This matches PaddleOCR's resize_norm_img function:
    1. Resize maintaining aspect ratio (height fixed)
    2. Normalize to [-1, 1] range
    3. Transpose to CHW format
    4. Pad to target width

    Args:
        img: Input image in BGR format (OpenCV default)
        image_shape: Target shape as (C, H, W)
        padding: Whether to pad to fixed width

    Returns:
        Tuple of (normalized image, valid_ratio)
    """
    imgC, imgH, imgW = image_shape
    h, w = img.shape[:2]

    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
    else:
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

    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def ctc_decode(probs: np.ndarray, characters: list[str]) -> tuple[str, float]:
    """Decode CTC output to text.

    Applies greedy CTC decoding:
    1. Argmax to get most likely character at each timestep
    2. Remove consecutive duplicates
    3. Remove blank tokens (index 0)
    4. Map indices to characters

    Args:
        probs: Model output probabilities of shape (1, T, num_classes)
               Note: PP-OCRv5 model outputs softmax probabilities, not logits
        characters: Character dictionary

    Returns:
        Tuple of (decoded text, average confidence)
    """
    # Model already outputs softmax probabilities
    probs = probs[0]  # Shape: (T, num_classes)

    # Greedy decode: take argmax at each timestep
    indices = np.argmax(probs, axis=1)  # Shape: (T,)
    confidences = np.max(probs, axis=1)  # Shape: (T,)

    # CTC decode: remove blanks and consecutive duplicates
    decoded_chars = []
    decoded_confs = []
    prev_idx = -1

    for i, idx in enumerate(indices):
        if idx != 0 and idx != prev_idx:  # Not blank and not duplicate
            if idx < len(characters):
                decoded_chars.append(characters[idx])
                decoded_confs.append(confidences[i])
        prev_idx = idx

    text = "".join(decoded_chars)
    avg_conf = float(np.mean(decoded_confs)) if decoded_confs else 0.0

    return text, avg_conf


class ONNXRecognizer:
    """ONNX-based text recognition model."""

    def __init__(self, model_path: str, dict_path: str,
                 image_shape: tuple[int, int, int] = (3, 48, 320),
                 verbose: bool = True):
        """Initialize the recognizer.

        Args:
            model_path: Path to ONNX model file
            dict_path: Path to character dictionary file
            image_shape: Model input shape as (C, H, W)
            verbose: Whether to print model info
        """
        self.image_shape = image_shape
        self.characters = load_character_dict(dict_path)

        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if verbose:
            print(f"Loaded ONNX model: {model_path}")
            print(f"Character dictionary size: {len(self.characters)}")
            print(f"Input name: {self.input_name}")
            print(f"Output name: {self.output_name}")
            print(f"Using providers: {self.session.get_providers()}")

    def predict(self, image_path: str) -> tuple[str, float]:
        """Run recognition on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (recognized text, confidence)
        """
        # Read image in BGR format
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict_array(img)

    def predict_array(self, img: np.ndarray) -> tuple[str, float]:
        """Run recognition on a numpy array image.

        Args:
            img: Input image as numpy array in BGR format

        Returns:
            Tuple of (recognized text, confidence)
        """
        # Preprocess
        norm_img, valid_ratio = resize_norm_img(img, self.image_shape)

        # Add batch dimension
        input_tensor = norm_img[np.newaxis, :, :, :]  # Shape: (1, C, H, W)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        probs = outputs[0]  # Shape: (1, T, num_classes)

        # Decode
        text, confidence = ctc_decode(probs, self.characters)

        return text, confidence

    def predict_raw(self, img: np.ndarray) -> np.ndarray:
        """Run recognition and return raw probabilities.

        Args:
            img: Input image as numpy array in BGR format

        Returns:
            Raw probability output of shape (1, T, num_classes)
        """
        # Preprocess
        norm_img, valid_ratio = resize_norm_img(img, self.image_shape)

        # Add batch dimension
        input_tensor = norm_img[np.newaxis, :, :, :]  # Shape: (1, C, H, W)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]


class PaddleRecognizer:
    """Paddle Inference-based text recognition model."""

    def __init__(self, model_dir: str, dict_path: str,
                 image_shape: tuple[int, int, int] = (3, 48, 320),
                 verbose: bool = True):
        """Initialize the recognizer.

        Args:
            model_dir: Path to Paddle inference model directory
            dict_path: Path to character dictionary file
            image_shape: Model input shape as (C, H, W)
            verbose: Whether to print model info
        """
        import paddle.inference as paddle_infer

        self.image_shape = image_shape
        self.characters = load_character_dict(dict_path)

        # Find model files
        model_file = None
        params_file = None

        for f in os.listdir(model_dir):
            if f.endswith('.pdmodel'):
                model_file = os.path.join(model_dir, f)
            elif f.endswith('.json') and 'inference' in f:
                model_file = os.path.join(model_dir, f)
            elif f.endswith('.pdiparams'):
                params_file = os.path.join(model_dir, f)

        if model_file is None or params_file is None:
            raise ValueError(f"Could not find model files in {model_dir}")

        # Create Paddle inference config
        config = paddle_infer.Config(model_file, params_file)
        config.enable_memory_optim()
        config.disable_glog_info()

        # Try to use GPU if available
        try:
            config.enable_use_gpu(500, 0)
        except Exception:
            pass

        self.predictor = paddle_infer.create_predictor(config)

        # Get input/output handles
        input_names = self.predictor.get_input_names()
        self.input_handle = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(output_names[0])

        if verbose:
            print(f"Loaded Paddle model: {model_dir}")
            print(f"Character dictionary size: {len(self.characters)}")
            print(f"Input name: {input_names[0]}")
            print(f"Output name: {output_names[0]}")

    def predict(self, image_path: str) -> tuple[str, float]:
        """Run recognition on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (recognized text, confidence)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict_array(img)

    def predict_array(self, img: np.ndarray) -> tuple[str, float]:
        """Run recognition on a numpy array image.

        Args:
            img: Input image as numpy array in BGR format

        Returns:
            Tuple of (recognized text, confidence)
        """
        # Preprocess
        norm_img, valid_ratio = resize_norm_img(img, self.image_shape)

        # Add batch dimension
        input_tensor = norm_img[np.newaxis, :, :, :]  # Shape: (1, C, H, W)

        # Run inference
        self.input_handle.reshape(input_tensor.shape)
        self.input_handle.copy_from_cpu(input_tensor)
        self.predictor.run()
        probs = self.output_handle.copy_to_cpu()

        # Decode
        text, confidence = ctc_decode(probs, self.characters)

        return text, confidence

    def predict_raw(self, img: np.ndarray) -> np.ndarray:
        """Run recognition and return raw probabilities.

        Args:
            img: Input image as numpy array in BGR format

        Returns:
            Raw probability output of shape (1, T, num_classes)
        """
        # Preprocess
        norm_img, valid_ratio = resize_norm_img(img, self.image_shape)

        # Add batch dimension
        input_tensor = norm_img[np.newaxis, :, :, :]  # Shape: (1, C, H, W)

        # Run inference
        self.input_handle.reshape(input_tensor.shape)
        self.input_handle.copy_from_cpu(input_tensor)
        self.predictor.run()
        return self.output_handle.copy_to_cpu()


def get_image_files(path: str) -> list[str]:
    """Get list of image files from a path (file or directory).

    Args:
        path: Path to image file or directory

    Returns:
        List of image file paths
    """
    path = Path(path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = []
        for f in sorted(path.iterdir()):
            if f.suffix.lower() in image_extensions:
                files.append(str(f))
        return files
    else:
        raise ValueError(f"Path does not exist: {path}")


def verify_models(onnx_recognizer: ONNXRecognizer,
                  paddle_recognizer: PaddleRecognizer,
                  image_paths: list[str],
                  rtol: float = 1e-4,
                  atol: float = 1e-5) -> dict:
    """Verify ONNX model output matches Paddle model output.

    Args:
        onnx_recognizer: ONNX-based recognizer
        paddle_recognizer: Paddle-based recognizer
        image_paths: List of image paths to test
        rtol: Relative tolerance for numpy.allclose
        atol: Absolute tolerance for numpy.allclose

    Returns:
        Dictionary with verification results
    """
    results = {
        'total': len(image_paths),
        'text_match': 0,
        'prob_close': 0,
        'failures': [],
        'details': []
    }

    print(f"\nVerifying {len(image_paths)} images...")
    print("=" * 80)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            results['failures'].append((image_path, "Could not read image"))
            continue

        try:
            # Get predictions
            onnx_text, onnx_conf = onnx_recognizer.predict_array(img)
            paddle_text, paddle_conf = paddle_recognizer.predict_array(img)

            # Get raw probabilities
            onnx_probs = onnx_recognizer.predict_raw(img)
            paddle_probs = paddle_recognizer.predict_raw(img)

            # Check if text matches
            text_match = (onnx_text == paddle_text)
            if text_match:
                results['text_match'] += 1

            # Check if probabilities are close
            prob_close = np.allclose(onnx_probs, paddle_probs, rtol=rtol, atol=atol)
            if prob_close:
                results['prob_close'] += 1

            # Calculate max difference
            max_diff = np.max(np.abs(onnx_probs - paddle_probs))
            mean_diff = np.mean(np.abs(onnx_probs - paddle_probs))

            # Store details
            detail = {
                'image': Path(image_path).name,
                'onnx_text': onnx_text,
                'paddle_text': paddle_text,
                'onnx_conf': onnx_conf,
                'paddle_conf': paddle_conf,
                'text_match': text_match,
                'prob_close': prob_close,
                'max_diff': max_diff,
                'mean_diff': mean_diff
            }
            results['details'].append(detail)

            # Print result
            status = "OK" if text_match and prob_close else "DIFF"
            text_status = "==" if text_match else "!="
            print(f"[{status}] {Path(image_path).name}")
            print(f"      ONNX:   '{onnx_text}' (conf: {onnx_conf:.4f})")
            print(f"      Paddle: '{paddle_text}' (conf: {paddle_conf:.4f})")
            print(f"      Text: {text_status} | Max prob diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}")

        except Exception as e:
            results['failures'].append((image_path, str(e)))
            print(f"[ERR] {Path(image_path).name}: {e}")

    print("=" * 80)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total images: {results['total']}")
    print(f"  Text matches: {results['text_match']}/{results['total']} "
          f"({100*results['text_match']/results['total']:.1f}%)")
    print(f"  Prob close (rtol={rtol}, atol={atol}): {results['prob_close']}/{results['total']} "
          f"({100*results['prob_close']/results['total']:.1f}%)")
    if results['failures']:
        print(f"  Failures: {len(results['failures'])}")
        for path, err in results['failures']:
            print(f"    - {Path(path).name}: {err}")

    # Overall verdict
    all_match = (results['text_match'] == results['total'] and
                 results['prob_close'] == results['total'] and
                 len(results['failures']) == 0)

    print(f"\nVerdict: {'PASS - ONNX export matches Paddle' if all_match else 'DIFFERENCES FOUND'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ONNX inference for PP-OCRv5 recognition model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--dict", "-d",
        type=str,
        required=True,
        help="Path to character dictionary file"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        nargs="+",
        help="Path(s) to input image(s) for inference"
    )
    parser.add_argument(
        "--verify", "-v",
        type=str,
        help="Path to image or folder to verify ONNX vs Paddle outputs"
    )
    parser.add_argument(
        "--paddle-model", "-p",
        type=str,
        help="Path to Paddle inference model directory (required for --verify)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=48,
        help="Model input height (default: 48)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Model input width for padding (default: 320)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for probability comparison (default: 1e-4)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for probability comparison (default: 1e-5)"
    )

    args = parser.parse_args()

    # Validation
    if not args.image and not args.verify:
        parser.error("Either --image or --verify must be specified")

    if args.verify and not args.paddle_model:
        parser.error("--paddle-model is required when using --verify")

    image_shape = (3, args.height, args.width)

    # Verification mode
    if args.verify:
        print("Verification mode: comparing ONNX vs Paddle inference")
        print("-" * 60)

        # Load both models
        onnx_recognizer = ONNXRecognizer(args.model, args.dict, image_shape, verbose=True)
        print()
        paddle_recognizer = PaddleRecognizer(args.paddle_model, args.dict, image_shape, verbose=True)

        # Get image files
        image_paths = get_image_files(args.verify)
        if not image_paths:
            print(f"No images found in {args.verify}")
            return

        # Run verification
        verify_models(onnx_recognizer, paddle_recognizer, image_paths,
                      rtol=args.rtol, atol=args.atol)
        return

    # Normal inference mode
    onnx_recognizer = ONNXRecognizer(args.model, args.dict, image_shape)

    print("\nResults:")
    print("-" * 60)

    for image_path in args.image:
        try:
            text, confidence = onnx_recognizer.predict(image_path)
            print(f"{Path(image_path).name}: '{text}' (conf: {confidence:.4f})")
        except Exception as e:
            print(f"{Path(image_path).name}: ERROR - {e}")

    print("-" * 60)


if __name__ == "__main__":
    main()
