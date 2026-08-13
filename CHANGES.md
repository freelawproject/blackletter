# Change Log

## Coming up

The following changes are not yet released, but are code complete:

- Stop paying for bitonal per-page work that was being discarded, worth about 1.5x on a 942-page volume at 200 DPI. Three fixes in the renderer: each page is thresholded with `np.packbits` in a single pass instead of materialising a full 8-bit image for PIL to threshold a second time; the TIFF handed to `insert_image` is no longer CCITT G4 compressed, because MuPDF decodes whatever it is given and re-encodes the embedded image as `/FlateDecode`, so that pass cost time and produced a byte-identical PDF; and ICC colour management is disabled for the render. The first two leave the output untouched. The third does not, quite: colour management shifts the greyscale a *colour* page rasterises to, so a pixel landing within that shift of the threshold can flip. On the volume above that was under 0.1% of pixels with ink coverage matching to two decimal places, and greyscale scans — what this is normally pointed at — are unaffected. Anyone diffing a re-converted colour volume against an old one should expect a handful of pixels to move. The per-page logic also no longer exists in three places: the copy that lived inside the `_BITONAL_WORKER_SCRIPT` string literal is gone, and `tasks.bitonal_chunk` goes through the same `_render_bitonal_page` as everything else, so a fix to the renderer can no longer miss a path (#71)
- Remove the process pool `ocr.bitonal_convert` used to start on documents of 40 pages or more. It sized itself from `multiprocessing.cpu_count()`, which reports host cores rather than what the container was granted, so on an unrestricted pod it would take the whole node and starve everything scheduled beside it. Conversion now runs in the calling process, and a caller that wants it parallelised fans out over `tasks.bitonal_chunk` and merges with `tasks.merge_pdfs`, sized against a CPU budget only that caller knows (#71)
- Fix `tests/test_lazy_imports.py` against PyMuPDF 1.28, which prints a `fitz` deprecation banner to stdout on import. The helper compared a subprocess's entire stdout to an expected value, so anything printed at import time failed 7 of these tests on `main` as well; it now reads the last non-empty line, which is the value each script prints (#71)

## Current

0.2.0 (2026-08-04)

- **Breaking:** `scanner.scan()` no longer runs the ocrmypdf pre-pass on a PDF with no text layer. The `skip_ocr` parameter is replaced by `ocr`, defaulting to `False`, so callers that passed `skip_ocr=True` should drop the argument and callers that relied on the automatic pass should pass `ocr=True`. The geometry was the only reason the pass had to run early, and it no longer needs it (#68)
- **Breaking:** `Document.ocr_applied` now selects where `compute_redaction_rects` and `_build_full_redacted` measure from, and it means "prefer the page ink to these word boxes" rather than "skip tightening". It was previously honoured only by `split_opinions`; `compute_redaction_rects` ignored it and always used words, and `_build_full_redacted` hardcoded it to `False`. So a caller that sets it — which every `Document` rebuilt from a detections sidecar does, since they all pass `ocr_applied=True` — now gets ink-measured rects where it used to get word-measured ones. Unlike the `skip_ocr` rename this cannot raise: the rects change shape rather than the call breaking, so a consumer that post-processes them should re-check its own geometry against a real volume rather than assume the upgrade is inert. Pass `ocr_applied=False` on a PDF whose text layer is trustworthy (born digital, not our own OCR) to keep the word boxes (#68)
- Add `api.add_text_layer(paths, ...)`, which OCRs PDFs that already exist, in place, and skips files that are already searchable. This is the step to use for searchable *output*: run it over the redacted deliverables rather than OCRing the source first, and no time is spent on content that is about to be blacked out or on a text layer that `apply_redactions` then has to scrub. `blackletter process --text-layer` applies it to the full redacted PDF and the per-opinion files in `redacted/`; `--ocr` restores the old pre-pass over the source. Files are processed several at a time with ocrmypdf's own workers sharing the machine between them, which measured 4x faster than one file at a time on a batch of short opinions (each ocrmypdf run carries several seconds of fixed cost) and still faster on a page-heavy batch (#68)
- Expose the steps an application driving a review loop needs, instead of only the all-in-one calls a CLI wants. `validate.build_analysis` projects existing per-page OCR results into the analysis `build_issues` reads, so correcting one page number by hand no longer means re-OCRing a volume to refresh its findings; `validate()` now uses it too, rather than carrying its own copy. `build_issues` and `parse_expected_range` are public (the old underscored names still work). `api.build_redactions` combines this library's own `compute_rects`, `compute_margin_rects` and pairing output into the payload `generate` consumes, converting image pixels to points from each page's own dimensions and assigning opinion filenames. `process.page_body_covered` answers "is this page's body entirely rects" from the geometry, which is what extracting text and finding none used to mean, and no longer can on a PDF with no text layer (#68)
- Finish redaction and margin rects against the detections inside `compute_redaction_rects` and `compute_margin_rects`, rather than leaving each consumer to post-process the output. Headnote rects have their side edges snapped to their own `TEXT_COLUMN` box (never one across the page midpoint, which would move the rect to the other side, and never one whose width is implausible for the rect, since a box spanning two columns blacks out the facing column's text and a degenerate one collapses the rect entirely), are cut at each `HEADNOTE` detection inside them, and are then grown onto adjoining ink, in that order. The cut does not reduce coverage: the pieces still tile the rect, and the point of it is that each piece is then measured against its own ink rather than the whole column's. Outside-opinion masks are grown at the point they are built, since they come from the same column boxes. Margin strips are pulled back off any real detection they would cover, ignoring the noisy page-edge labels that define the content box in the first place (#68)
- Overdraw each applied redaction fill-only in `api.generate`, so a seam between a black rect and a white one cannot show a hairline. PyMuPDF has painted redactions as a fill plus a 1pt stroke straddling the edge, strokes last, and that was visible in real deliverables; the CLI path compensated for it and the `generate` API did not. It does not reproduce on PyMuPDF 1.26.7, so this is insurance against a version that strokes again rather than a fix for something currently visible (#68)
- Add `text_layer` and `ocr` parameters to `process()`, which previously built its argument namespace by hand and omitted both, so the Python entry point could not reach either flag the CLI offers (#68)
- Fix redaction and margin geometry on PDFs with no text layer, which previously failed silently rather than degrading: `_text_bottom` returned the top of its clip, so every headnote rect collapsed to zero height and was dropped, and `margins._text_bounds` returned nothing, so margin cleanup did nothing at all. Both now measure the page's ink (dark-pixel projections, new `blackletter.ink` module) when words are unavailable, so a bitonal PDF gets the same geometry a post-OCR one does. `Document.ocr_applied` is now honoured by `compute_redaction_rects` and `_build_full_redacted` as well, and means "prefer ink to these word boxes" rather than "skip tightening": our own OCR's positions are imprecise, while ink is measured from the pixels the OCR read (#68)
- Correct `TEXT_COLUMN` detections against the page ink wherever a `Document` is built, not only in `scan()` (`snap_text_columns_to_ink`, `snap_document_columns`). A caller that detects remotely, or reloads detections from a sidecar, holds boxes that were never corrected, and margin strips derived from them can be narrower than the page's own text. YOLO's column boxes land a median 0.4 pt (up to ~7 pt) inside the printed text, and three consumers read them — headnote rect x-bounds, the margin text band, and the outside-opinion masks — so a narrow box left the first or last character of every masked line behind. Each column may grow only to the middle of the gutter it shares with its neighbour, and an edge that runs the whole way to the growth limit is left where the detector put it (#68)
- Tighten margin cleanup with detection geometry: `compute_margin_rects` and `clean_margins` accept the detected `pages` and intersect the measured content box with the `TEXT_COLUMN` band and the header row, so one bleed-through mark in a corner no longer widens the content box and shrinks the strip on that side. The four strips are also reshaped — full-width above and below the body, side strips spanning the body rows only — which covers the same region while keeping a side strip away from a page number printed outside the columns. On a real volume this raised masked area 23% and junk ink covered 3.3x, and no ink inside a content detection is erased across 60 pages of a real volume, against 27,267 px of scanner artifacts that are. A side bound is only tightened past ink that reads as an artifact, near-solid or negligible in its own pixel columns; text-like ink means a column went undetected and the band is not to be trusted there (#68)
- `clean_margins` now computes its strips through `compute_margin_rects` instead of duplicating the geometry, so the two can no longer disagree (#68)

## Past

0.1.1 (2026-07-15)

- **Breaking (packaging):** stop bundling `small.pt` and `medium.pt` in the package (~72 MB smaller wheel). All three YOLO weights are now downloaded on demand from [freelawproject/blackletter-weights](https://huggingface.co/freelawproject/blackletter-weights) via `ensure_weights`, the same way `large.pt` already was; `large.pt` moves from `flooie/blackletter-large` to the same consolidated repo. Downloads are pinned to a specific commit of the weights repo so a compromised repo can't serve different binaries. Downloading requires `huggingface_hub` (installed with `blackletter[detect]`) (#59)
- **Breaking (packaging):** move `python-doctr[torch]` out of the `detect` extra into a new `refine` extra, so `blackletter[detect]` / `blackletter[analyze]` no longer install docTR (~200 MB of dependencies). docTR is only used by the line-level headnote-refinement pass in `refine.py`; consumers that call redaction with `skip_doctr=True` (e.g. the scanning web/daemon image) never invoke it. Installs that run refinement — including the `process` CLI command — must now add `[refine]` (e.g. `blackletter[analyze,refine]`). With docTR missing, the refinement pass raises a clear `ImportError` pointing at `pip install blackletter[refine]` / `skip_doctr=True` (#64)


0.1.0 (2026-07-10)

- **Breaking (packaging):** split the heavy inference stack out of the base dependencies into optional extras. `ultralytics` + `python-doctr[torch]` + `huggingface_hub` now install via `blackletter[detect]`; `blackletter[analyze]` adds PaddleOCR on top of `detect` (the analyze pipeline runs YOLO too). The base install (`pip install blackletter`) keeps only pairing, redaction, margins, validation, and OCR. Base stays on `opencv-python` (not `-headless`): the `detect`/`analyze` extras pull `opencv-python` transitively via ultralytics and python-doctr, and both cv2 distributions own the same top-level `cv2` package, so a headless base would double-install a conflicting cv2 whenever an extra is present (#61). Consumers that run detection locally must install `blackletter[detect]` (or `[analyze]`); those that offload detection can stay lean and call redaction helpers with `skip_doctr=True`. To keep every advertised install path importable: the CLI's top-level `ultralytics` import moved into `cmd_draw`, and `compute_rects` / `pair_and_compute_rects` now propagate `skip_doctr` so lean-base callers can compute rects from remote detections without `doctr`
- Flag every copy of a repeated page number as `duplicate` in `page_map`, not just the 2nd and later copies, so per-page duplicate markers match the `duplicate_page` issue's page list. Missing-page placeholders still anchor on the first copy (#55)
- Add an optional `progress_callback` to `api.ocr()` that reports `(pages_done, total_pages)` as Tesseract completes pages, so consumers can surface OCR progress without copying the whole function. Default behavior (no callback, no progress bar) is unchanged (#60)

0.0.13 (2026-06-18)

- Fix `validate()` flagging real numbered pages as duplicates when a PDF starts with unnumbered front matter: pages with no detected number fell back to a `logical = pdf_page` placeholder that collided with the real page numbers. Only genuinely detected single page numbers now participate in `page_map` duplicate detection (#53)

0.0.12 (2026-05-29)

- Fix `TypeError` in `_ocr_crop_multi`'s page-number validation when `exp_start` is set but `exp_end` is `None`, which crashed `analyze_pdf` on scans with a known start page but unknown end page (#50)

0.0.11 (2026-05-28)

- Close every `fitz.Document` via `with fitz.open(...)` in `margins.py`, `api.py`, and `process.py` so exceptions no longer leak the open (mmap-backed) Document, fixing the runaway memory growth seen in downstream scanning web pods (#47)

0.0.10 (2026-05-14)

- Lazy-load `ultralytics`/`torch` so importing `blackletter` (or any submodule) no longer pulls in the GPU stack. CPU-only consumers (e.g. scanning daemons running with RunPod) save ~500 MB to 1 GB of resident memory; YOLO-using code paths are unchanged (#44)
- Remove `masked/` output entirely (per-opinion masked PDFs are no longer generated). Replaced with an opt-in `llm/` directory: one PDF per source page sliced from the fully redacted document, with an invisible `<--CASEEND-->` text stamp (`render_mode=3`) on every redacted Key-icon location so downstream LLM passes can detect opinion boundaries. Enable with `--llm` on `blackletter process` or `llm=True` on `api.generate()` (#45)
- Drop the `WHITE_IN_MASKED` override from `api.generate._apply_page`: `PAGE_HEADER` and `STATE_ABBREVIATION` now use the fill colour from `redactions.json` in all output modes instead of being forced black outside masked mode (#45)
- Delete `_build_masked_opinions` and `_delete_headnote_pages` (process.py); collapse `_apply_page`'s `mode` parameter from `{full, redacted, masked}` to `{full, redacted}` (api.py) (#45)

0.0.9 (2026-04-29)

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

0.0.8 (2026-04-01)

- Update api generate file names

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

0.0.1 - Initial release
