# Blackletter

A reference to blackletter law, this tool removes potentially copyrighted material from legal case law PDFs. This fulfills our goal of respecting any intellectual property rights others may have while making it possible to digitize and publish case law. This is essential to our mission of making case law accessible to all — a prerequisite for meaningful participation in our democracy.

Proprietary annotations removed from judicial opinions include headnotes, captions, and key cites.


## Installation

```bash
pip install blackletter
```

Or install from source:
```bash
git clone https://github.com/freelawproject/blackletter
cd blackletter
pip install -e .
```

## Quick Start

**Command line:**
```bash
blackletter process path/to/volume.pdf --reporter f3d --volume 952 --first-page 1 --output output/
```

**Python:**
```python
from blackletter import process

process(
    "path/to/volume.pdf",
    "output/",
    reporter="f3d",
    volume="952",
    first_page=1,
)
```

This runs the full pipeline: OCR (if needed), YOLO detection, page number extraction, opinion splitting, and redaction — all in one pass.

## How It Works

The `process` command runs a single-pass pipeline:

1. **OCR** (if needed): Detects image-only PDFs, downsamples pages, and adds a text layer via ocrmypdf/tesseract
2. **Detection**: Runs a YOLO model to identify proprietary elements (headnotes, captions, key cites, brackets, etc.) and structural elements (page numbers, dividers, footnotes)
3. **Page Numbers**: Extracts and validates page numbers using OCR on detected regions
4. **Opinion Pairing**: Matches case captions to key icons to identify opinion boundaries
5. **Splitting & Redaction**: Produces three output variants per opinion:
   - **Unredacted**: Raw opinion pages extracted from the source
   - **Redacted**: Potentially copyrighted content (headnotes, brackets, key icons) blacked out; non-opinion content whited out
   - **Masked**: Optimized for LLM ingestion — only the opinion text is visible

Additionally produces:
- A full redacted copy of the entire document
- Extracted case law images (charts, photos, etc.) as PNGs
- A `detections.json` export of all YOLO detections for review tooling

## Models

Blackletter bundles three YOLO models, selected via CLI flags:

| Flag | File | Classes | Description |
|------|------|---------|-------------|
| *(default)* | `small.pt` | 14 | Bundled — fast, handles most cases |
| `--medium` | `medium.pt` | 17 | Bundled — better structural detection |
| `--large` | `large.pt` | 21 | Downloaded on first use from Hugging Face — highest accuracy, detects additional elements (editorial, judges, docket, court, citation, date) |

The large model is hosted at [flooie/blackletter-large](https://huggingface.co/flooie/blackletter-large) and is downloaded automatically to `blackletter/models/large.pt` the first time `--large` is used.

## Command Line Options

### Process Command

```
blackletter process PDF [OPTIONS]

Positional Arguments:
  pdf                       Path to the source PDF

Options:
  --reporter STR            Reporter abbreviation (e.g. f3d, a3d)
  --volume STR              Volume number
  --first-page INT          Page number of the first page in the PDF (default: 1)
  -o, --output PATH         Base output directory (required)
  --model PATH              Path to custom YOLO model weights
  --medium                  Use the medium model (17 classes)
  --large                   Use the large model (21 classes, auto-downloaded)
  --footnotes               Extract footnotes into separate PDFs
  --unredacted              Also generate unredacted opinion PDFs
  --no-shrink               Skip downsampling (default: shrink to ~148 KB/page)
  --optimize {0,1,2,3}      ocrmypdf optimization level (default: 1)
  --bitonal                 Convert to 1-bit B&W before processing (for already-bitonal scans)
  --detect-only             Stop after detection and pairing — no PDFs written (Phase 1 only)
```

### Validate Command

QA tool that checks a PDF's page number sequence for missing, duplicate, or misnumbered pages. Uses YOLO to locate page number regions, then PaddleOCR to read them, with Tesseract and GLM-OCR as fallbacks.

```bash
blackletter validate path/to/volume.pdf
blackletter validate path/to/volume.pdf --first-page 100 --last-page 500
blackletter validate path/to/volume.pdf --json
```

If the filename follows the convention `reporter.volume.first.last.pdf` (e.g. `sct.143.1.888.pdf`), the expected page range is inferred automatically.

Features:
- Parallel OCR across multiple workers
- Auto-correction of consistent OCR misreadings (e.g. systematic off-by-800 errors)
- Detection of gaps, duplicates, backwards jumps, and page ranges (e.g. "31-32")
- Structural checks for blank pages and orientation changes

Requires optional dependencies: `pip install blackletter[analyze]`

### Draw Command

Visualize YOLO detections on a PDF — useful for debugging model output:

```bash
blackletter draw path/to/volume.pdf --output annotated.pdf
blackletter draw path/to/volume.pdf --output annotated.pdf --labels CASE_CAPTION KEY_ICON HEADNOTE
blackletter draw path/to/volume.pdf --output annotated.pdf --large
```

## Output Structure

```
output/<reporter>/<volume>/<first-page>/
    <reporter>.<volume>.<first>.<last>.pdf   # OCR'd/processed source PDF
    <reporter>.<volume>.redacted.pdf         # Full redacted document

    detections.json       # All YOLO detections (label, bbox, confidence per page)
    pages_meta.json       # Column bounds and midpoints per page
    opinions.json         # Opinion pairs with outside-opinion rects
    redaction_rects.json  # Precomputed redaction rectangles (used by review UI)
    margin_rects.json     # Margin cleanup rectangles

    images/               # Extracted case law images (PNGs)
    unredacted/           # Individual opinion PDFs (raw, no redaction)
    redacted/             # Individual opinion PDFs (copyrighted content redacted)
    masked/               # Individual opinion PDFs (for LLM ingestion)
```

The JSON files are designed for use with a review UI — they allow manual inspection and adjustment of detections and redaction boundaries before final output is committed.

## Detection Labels

Labels detected across all models (availability depends on model size):

| Label | Models | Description |
|-------|--------|-------------|
| KEY_ICON | all | West key cite icons |
| DIVIDER | all | Opinion section dividers |
| PAGE_HEADER | all | Running headers |
| CASE_CAPTION | all | Opinion title/parties |
| FOOTNOTES | all | Footnote sections |
| HEADNOTE_BRACKET | all | Bracketed headnote markers |
| CASE_METADATA | all | Court, date, counsel info |
| CASE_SEQUENCE | all | Docket/case sequence numbers |
| PAGE_NUMBER | all | Page numbers |
| STATE_ABBREVIATION | all | State abbreviation markers |
| IMAGE | all | Photos, charts, diagrams |
| HEADNOTE | all | Headnote text |
| BACKGROUND | all | Background/procedural history region |
| SYLLABUS | all | Supreme Court syllabus sections |
| EDITORIAL | medium, large | Editorial notes |
| JUDGES | medium, large | Judge name blocks |
| TEXT_COLUMN | medium, large | Column boundaries |
| DOCKET | large | Docket number regions |
| DATE | large | Decision date regions |
| COURT | large | Court name regions |
| CITATION | large | Reporter citation regions |

## Margin Cleanup

After redaction, Blackletter automatically white-outs scan artifacts in page margins using the PDF text layer to find content boundaries. Pages with narrow text spans (appendices, image pages) are skipped automatically.

## Requirements

- Python 3.12+
- Tesseract OCR (for image-only PDFs)

Install tesseract:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

## License

GNU Affero General Public License v3

## Contributing

Contributions welcome!
