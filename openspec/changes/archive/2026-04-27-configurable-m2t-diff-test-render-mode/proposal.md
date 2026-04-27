## Why

MotionFix `m2t_diff` test visualization currently hardcodes fast rendering, even though the repository already has a slower higher-fidelity single-motion render path. Users need a config-driven way to choose fast or slow rendering during `test.py` runs without mixing cache artifacts across render modes.

## What Changes

- Add a test-time config option to choose the `m2t_diff` visualization render method.
- Extend `m2t_diff` test visualization so source, target, and side-by-side videos all honor the configured render method.
- Separate fast and slow render caches so changing render mode never overwrites or reuses artifacts from the other mode.
- Document the new config option and its runtime trade-off.

## Capabilities

### New Capabilities

### Modified Capabilities
- `two-motion-difference-test-visualization`: test visualization must allow choosing the render method for all exported videos and keep fast and slow caches isolated.

## Impact

- Affected code: `configs/default.yaml`, MotionFix test visualize configs, `motGPT/models/base.py`, `motGPT/utils/render_utils.py`, and `README.md`
- Affected outputs: `results/.../samples_<TIME>/` continue to use the same filenames, while MotionFix render cache paths become render-method-specific for source, target, and side-by-side videos
