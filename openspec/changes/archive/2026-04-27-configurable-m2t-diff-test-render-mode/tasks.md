## 1. Config And Render Plumbing

- [x] 1.1 Add a `TEST.VISUALIZE_RENDER_METHOD` config default and set it in the MotionFix test visualization example config.
- [x] 1.2 Thread the configured render method through the shared `m2t_diff` test artifact save path in `motGPT/models/base.py`.

## 2. Render Cache And Side-By-Side Rendering

- [x] 2.1 Update `motGPT/utils/render_utils.py` so MotionFix source, target, and side-by-side cache directories are all separated by render method.
- [x] 2.2 Implement slow side-by-side rendering so `source_target.mp4` honors the configured render method.

## 3. Validation And Docs

- [x] 3.1 Update `README.md` to document the new test visualization render-method option and the fast versus slow cache separation.
- [x] 3.2 Run targeted validation, including `uv run python -m py_compile test.py motGPT/models/base.py motGPT/utils/render_utils.py` and a config load check for the MotionFix visualization config.
