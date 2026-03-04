from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

import pandas as pd


class CSVLogger:
    """
    Simple CSV logger that accumulates rows and writes to disk.
    """
    def __init__(self, out_csv_path: str):
        self.out_csv_path = out_csv_path
        self.rows: List[Dict[str, Any]] = []

        out_dir = os.path.dirname(out_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        df = pd.DataFrame(self.rows)
        df.to_csv(self.out_csv_path, index=False)

    def close(self) -> None:
        self.flush()
