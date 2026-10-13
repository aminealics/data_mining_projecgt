from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from core.synthetic import make_synthetic


def main() -> None:
    out = ROOT / "data" / "test_patient_data.csv"
    out.parent.mkdir(exist_ok=True)
    df = make_synthetic(n=2000, seed=42)
    df.to_csv(out, index=False)
    print(f"Created {out}")
    print(df.shape)
    print(df["readmitted"].value_counts().to_dict())


if __name__ == "__main__":
    main()
