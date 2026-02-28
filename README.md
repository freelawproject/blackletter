# Blackletter

Remove copyrighted material from legal case law PDFs.

A reference to blackletter law, this tool removes proprietary annotations from judicial opinions—specifically, headnotes, captions, key cites, and other copyrighted materials—while preserving the authentic opinion text.

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
2. **Detection**: Runs a YOLO model to identify copyrighted elements (headnotes, captions, key cites, brackets, etc.) and structural elements (page numbers, dividers, footnotes)
3. **Page Numbers**: Extracts and validates page numbers using OCR on detected regions
4. **Opinion Pairing**: Matches case captions to key icons to identify opinion boundaries
5. **Splitting & Redaction**: Produces three output variants per opinion:
   - **Unredacted**: Raw opinion pages extracted from the source
   - **Redacted**: Copyrighted content (headnotes, brackets, key icons) blacked out; non-opinion content whited out
   - **Masked**: Optimized for LLM ingestion — only the opinion text is visible

Additionally produces:
- A full redacted copy of the entire document
- A verification report with detection stats and page number mappings
- Extracted case law images (charts, photos, etc.) as PNGs

## Command Line Options

```
blackletter process PDF [OPTIONS]

Positional Arguments:
  pdf                       Path to the source PDF

Options:
  --reporter STR            Reporter abbreviation (e.g. f3d, a3d)
  --volume STR              Volume number
  --first-page INT          Page number of the first page in the PDF (default: 1)
  -o, --output PATH         Base output directory (required)
  --model PATH              Path to YOLO model weights (default: bundled run_9.pt)
  --footnotes               Extract footnotes into separate PDFs
  --no-unredacted           Skip generating unredacted opinion PDFs
  --no-shrink               Skip downsampling (default: shrink to ~148 KB/page)
  --optimize {0,1,2,3}      ocrmypdf optimization level (default: 1)
```

### Draw Command

For debugging YOLO detections:
```bash
blackletter draw path/to/volume.pdf --output annotated.pdf
blackletter draw path/to/volume.pdf --output annotated.pdf --labels CASE_CAPTION KEY_ICON HEADNOTE
```

## Output Structure

```
output/<reporter>/<volume>/<first-page>/
    verify.txt                          # Detection stats and page number report
    <reporter>.<volume>.redacted.pdf    # Full redacted document
    images/                             # Extracted images (PNGs)
    unredacted/                         # Individual opinion PDFs (raw)
    redacted/                           # Individual opinion PDFs (copyrighted content redacted)
    masked/                             # Individual opinion PDFs (for LLM ingestion)
```

## Detection Labels

The YOLO model detects 13 element types:

| Label | Description |
|-------|-------------|
| KEY_ICON | West key cite icons (copyrighted) |
| DIVIDER | Opinion section dividers |
| PAGE_HEADER | Running headers (copyrighted) |
| CASE_CAPTION | Opinion title/parties |
| FOOTNOTES | Footnote sections |
| HEADNOTE_BRACKET | Bracketed headnote markers (copyrighted) |
| CASE_METADATA | Court, date, counsel info |
| CASE_SEQUENCE | Docket/case sequence numbers |
| PAGE_NUMBER | Page numbers |
| STATE_ABBREVIATION | State abbreviation markers |
| IMAGE | Photos, charts, diagrams |
| HEADNOTE | Headnote text (copyrighted) |
| BACKGROUND | Background/procedural history |

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