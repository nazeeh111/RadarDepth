# Verification

Executed on 2026-09-22. Python 3.12.13, PyTorch 2.14.0, NumPy 1.26.4 on macOS ARM64.

## Passed

- 28 tracked numerical-source, firmware, configuration, and media files remained byte-identical to the baseline; 23 original Python files passed syntax parsing. [File hashes](source-verification.json).
- The new command dispatch test passed, checking argument preservation, paths containing spaces, expected working directories, and child exit-code propagation for every exposed command. Dispatch was mocked to avoid launching hardware routes.
- The complete DRNet3D forward path with segmentation enabled and disabled produced byte-identical output arrays against the baseline for a seeded random 128×8×8 radar tensor and random model weights. Depth, confidence, spread, and enabled mask outputs were finite. [Kernel evidence](kernel-verification.json).
- New command help works without importing optional hardware or model dependencies.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/verify_sources.py --baseline-dir /path/to/previous-checkout
python scripts/verify_kernels.py --baseline-dir /path/to/previous-checkout
```

Parity commands take an explicit separate prior checkout; they do not depend on unpublished historical Git objects. Kernel dependencies are separated from the full application requirements. Source syntax checks do not prove that optional dependencies or hardware work.

## Not executed

ColoRadar dataset conversion, training, checkpoint inference, Open3D/Rerun visualization, and real depth accuracy were not executed. The full application dependency stack was not installed; the neural-network kernel requires only the separately documented verification dependencies.

These are bounded source and synthetic-runtime checks, not universal correctness or research-replication claims. Original numerical code, calibration, and recorded scientific images were preserved.
