# Change Log

## Current

0.0.6 (2026-04-01)

- Re-release: v0.0.5 was tagged before #27 was merged

## Past

0.0.5 (2026-04-01)

- Skip multiprocessing Pool for single worker (#27)

0.0.4 (2026-03-20)

- Feature 'Add API and remove threading'

0.0.3 (2026-03-20)

- Fix `blackletter validate` crashing with FileNotFoundError when large.pt is missing (now auto-downloads from Hugging Face)
- Fix `analyze.py` referencing old model name `analyze.pt` instead of `large.pt`

0.0.2 (2026-03-20)

- Add small/medium/large model tiers; large model auto-downloads from Hugging Face on first use
- Expand Label enum to 21 classes to support large model (run_59)
- Add `--medium` and `--large` CLI flags to `process` and `draw` commands
- Add `blackletter validate` command for page number QA with auto-correction, gap/duplicate detection, and JSON output
- Use BACKGROUND detection as headnote redaction start when available (large/medium models)
- Add key icon aspect ratio filter (1.5–4.0 width:height) applied to both opinion pairing and redaction
- Raise KEY_ICON confidence threshold to 0.90
- Redact EDITORIAL label when detected
- Fix masked PDF page deletion regression (use block-level rects for coverage check)
- Add margin cleanup: white-out scan artifacts beyond text content area, skipping narrow-text pages
- Add docTR line-level refinement of headnote redaction rects
- CASE_SEQUENCE redactions now clip around CASE_CAPTION and inset by 3pt to avoid over-redaction
- Export detections.json, pages_meta.json, opinions.json, redaction_rects.json, margin_rects.json for review tooling
- Remove defunct verify.txt pipeline
- Rename bundled model from best.pt to small.pt
- Update README with full documentation of all commands, models, and output files

## Past

0.0.1 - Initial release
