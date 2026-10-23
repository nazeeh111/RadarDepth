# Verification

Executed on 2026-09-22. Python 3.12.13, PyTorch 2.14.0, NumPy 1.26.4 on macOS ARM64.

## Historical checks: 2026-09-22

- 28 tracked numerical-source, firmware, configuration, and media files remained byte-identical to the baseline; 23 original Python files passed syntax parsing. [File hashes](source-verification.json).
- The new command dispatch test passed, checking argument preservation, paths containing spaces, expected working directories, and child exit-code propagation for every exposed command. Dispatch was mocked to avoid launching hardware routes.
- The complete DRNet3D forward path with segmentation enabled and disabled produced byte-identical output arrays against the baseline for a seeded random 128×8×8 radar tensor and random model weights. Depth, confidence, spread, and enabled mask outputs were finite. [Kernel evidence](kernel-verification.json).
- New command help works without importing optional hardware or model dependencies.

## Current helper corrections: 2026-10-23

Two retained numerical files now intentionally differ from the original: `utils/velocity.py` samples Doppler by depth-derived range and keeps point/confidence rows aligned; `utils/radar.py` uses max-minus-min normalization and finite lower-endpoint colors for constant arrays. [Intent and original/current hashes](intentional-source-changes.json) explicitly record these changes. The historical [source manifest](source-verification.json) is unchanged.

The current source gate checks the 28 original paths pinned by that manifest: 26 remain byte-identical, and the two helper files must match their separately recorded current hashes. The explicit baseline must match every pinned original hash. Added facade, tests and verification documents are outside the original-file parity count. Any further original-file change fails the gate unless separately reviewed.

Eight synthetic helper/dispatch tests passed with Python 3.12.14 and bundled NumPy 2.3.5. They use the actual projection and turbo interpolation with invented calibration metadata. They cover range/azimuth separation, shared confidence masks, range clipping, near-zero/empty depth, finite constants, signed min/max normalization and palette endpoints. Empty intensity retains its existing NumPy `ValueError`; no new invalid-input policy was added. These checks do not execute trained inference, GTSAM, GPU or a dataset. The historical random-weight network check above was not rerun for this helper correction.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/verify_sources.py --baseline-dir /path/to/previous-checkout
python scripts/verify_kernels.py --baseline-dir /path/to/previous-checkout
```

Parity commands take an explicit separate prior checkout; they do not depend on unpublished historical Git objects. Kernel dependencies are separated from the full application requirements. Source syntax checks do not prove that optional dependencies or hardware work.

## Not executed

ColoRadar dataset conversion, training, checkpoint inference, Open3D/Rerun visualization, and real depth accuracy were not executed. The full application dependency stack was not installed; the neural-network kernel requires only the separately documented verification dependencies.

These are bounded source and synthetic-runtime checks, not universal correctness or research-replication claims. The historical checks preserved original numerical code. The current two helper corrections are explicitly recorded above; all other pinned original files, calibration and recorded scientific images remain unchanged.
