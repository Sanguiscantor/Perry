"""Regenerate artifacts/data/microstructure/manifest.json by scanning CSVs."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts" / "data" / "microstructure"
ART.mkdir(parents=True, exist_ok=True)
manifest = {"generated_at": None, "files": []}
for p in sorted(ART.glob("*.csv")):
    try:
        rows = int(pd.read_csv(p).shape[0])
    except Exception:
        rows = None
    manifest["files"].append({"path": str(p), "rows": rows})
from datetime import datetime, timezone
manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
with open(ART / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote {ART / 'manifest.json'}")
