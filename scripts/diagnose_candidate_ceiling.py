#!/usr/bin/env python3
"""Convenience wrapper for the read-only V1 candidate ceiling audit."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from error_analysis_grid_v2 import main as audit_main


if __name__ == "__main__":
    # Keep one interface so MacBERT and WWM can be diagnosed together.
    audit_main()
