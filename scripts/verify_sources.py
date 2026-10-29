"""Verify pinned original files and explicitly recorded source repairs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-dir", required=True, type=Path)
args = parser.parse_args()
manifest_path = ROOT / "docs/source-verification.json"
manifest = json.loads(manifest_path.read_text())
amendment = json.loads((ROOT / "docs/intentional-source-changes.json").read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == amendment["historical_manifest_sha256"], "Historical source manifest changed"
modified = amendment["modified_paths"]
assert set(modified) == {
    "utils/radar.py", "utils/velocity.py", "utils/pairing.py", "utils/__init__.py",
    "dataloader/dataset.py", "generate_dataset/numpy_to_trainset.py",
    "generate_dataset/raw_dataset_to_numpy.py", "train.py", "eval.py",
    "networks/DepthRefinement/network_parts.py",
}, "Unexpected source amendment paths"
unchanged = {}
intentional = {}
python_files = 0
for relative, original_hash in sorted(manifest["sha256"].items()):
    source = args.baseline_dir / relative
    current = ROOT / relative
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash, f"Baseline differs from pinned original: {relative}"
    current_hash = hashlib.sha256(current.read_bytes()).hexdigest()
    if relative in modified:
        record = modified[relative]
        assert record["original_sha256"] == original_hash, f"Original amendment hash differs: {relative}"
        assert current_hash == record["current_sha256"], f"Unrecorded source change: {relative}"
        intentional[relative] = record
    else:
        assert current_hash == original_hash, f"Original file changed: {relative}"
        unchanged[relative] = current_hash
    if current.suffix == ".py":
        ast.parse(current.read_text(), filename=relative)
        python_files += 1
assert unchanged, "No original files checked"
print(json.dumps({"unchanged_source_config_media_files": len(unchanged),
                  "intentionally_modified_files": intentional,
                  "python_files_parsed": python_files, "sha256": unchanged}, indent=2))
