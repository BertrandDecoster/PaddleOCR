#!/usr/bin/env python3
"""CLI entry point for OCR pipeline."""

try:
    # When running as a module from parent directory: python -m exported_ocr_pipeline
    from .ocr_pipeline import main
except ImportError:
    # When running from within the directory: python -m ocr_pipeline
    from ocr_pipeline import main

if __name__ == "__main__":
    main()
