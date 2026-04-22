# Change Log

## Coming up

The following changes are not yet released, but are code complete:

- Add rST docstrings (`:param:` / `:returns:`) across all public and internal functions in `api.py`, `margins.py`, `models.py`, `analyze.py`, `scanner.py`, and `validate.py`
- Add `Callable` type hints for `progress_callback` parameters in `api.py`, `analyze.py`, `scanner.py`, and `validate.py`
- Fix resource leak in `api.ocr()` where `fitz.open()` was never closed
- Remove dead `list[dict]` type from `api.build_redacted()` rects parameter
- Fix stale module-level usage examples in `api.py`
- Remove unnecessary import aliases (`_re`, `_Counter`, `_fitz`, `_I`) across `api.py` and `analyze.py`
- Remove em dashes from comments and strings project-wide
- Add lazy per-page word cache for `_tighten_to_text`, `_text_bottom`, and `_text_x_bounds`, replacing repeated `fitz_page.get_text()` calls (#35)
- Replace two `document.by_label()` calls with a single pass over page detections in `_pair_opinions()`, reducing from 3 sorts to 1 (#36)
- Replace per-opinion detection scanning with a single pre-sorted list and bisect slicing in `_build_full_redacted` (#37)
- Eliminate temp PNG file writes during OCR crop processing, reducing I/O overhead and preventing leaked files in `/tmp` on crashes (#38)
- Add check changelog action (#40)
- Add new helper to download weights (#42)
- Consolidate model download logic through `ensure_weights`: `detect()` now downloads missing weights instead of silently skipping, and `cli.py` and `analyze.py` no longer duplicate the Hugging Face download code (#42)

## Current

0.0.8 (2026-04-01)

- Update api generate file names

## Past

0.0.7 (2026-04-01)

- Simplify API call for generating file


0.0.6 (2026-04-01)

- Re-release: v0.0.5 was tagged before #27 was merged

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
