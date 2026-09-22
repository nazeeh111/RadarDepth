"""Compare seeded CPU depth-network outputs without data or trained checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(root))
    from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
    record = {}
    for segment in (False, True):
        torch.manual_seed(42)
        model = DRNet3D(DEPTH_VAL, segment=segment).eval()
        sample = torch.randn(1, 1, 128, 8, 8)
        with torch.no_grad():
            result = model(sample)
        output = {}
        for key, value in result.items():
            if value is None:
                output[key] = None
            else:
                array = value.numpy()
                assert np.isfinite(array).all()
                output[key] = {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        record[f"segment_{segment}"] = output
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(args.snapshot.resolve()), sort_keys=True))
    else:
        if not args.baseline_dir:
            parser.error("--baseline-dir is required")
        original = subprocess.check_output([sys.executable, __file__, "--snapshot", str(args.baseline_dir.resolve())])
        branded = subprocess.check_output([sys.executable, __file__, "--snapshot", str(ROOT)])
        assert original == branded, "Network outputs differ"
        print(json.dumps({"exact_kernel_parity": True, "scope": "CPU random weights and synthetic 128x8x8 radar tensor; not trained depth accuracy", "outputs": json.loads(branded)}, indent=2))
