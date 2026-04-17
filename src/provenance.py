"""Content provenance helpers for pipeline stage outputs.

Each pipeline stage stamps its output CSV with short hashes of the
input files and config module that produced it. Downstream stages
can then verify that their inputs match the expected upstream state.
"""

import hashlib
import json
from pathlib import Path
from typing import List

import pandas as pd


def file_hash(path: Path) -> str:
    """Return a short sha256 hex digest for the file content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stamp_provenance(
    output_df: pd.DataFrame,
    input_paths: List[Path],
    config_path: Path,
) -> pd.DataFrame:
    """Add _provenance columns tracking the inputs that produced this frame.

    The columns are metadata, not analysis data. They let a downstream
    consumer answer 'was this generated from the current config?' with
    a grep instead of re-running the pipeline.
    """
    input_hashes = {p.name: file_hash(p) for p in input_paths if p.exists()}
    config_hash = file_hash(config_path) if config_path.exists() else "missing"
    output_df["_provenance_inputs"] = json.dumps(input_hashes)
    output_df["_provenance_config"] = config_hash
    return output_df
