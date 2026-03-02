#!/usr/bin/env python
"""Compatibility launcher for training after moving the implementation to src/train.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "src" / "train.py"
    runpy.run_path(str(script_path), run_name="__main__")
