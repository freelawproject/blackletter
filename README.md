# Blackletter

Remove copyrighted material from legal case law PDFs.

A reference to blackletter law, this tool removes proprietary annotations from judicial opinions—specifically, headnotes, captions, key cites, and other copyrighted materials—while preserving the authentic opinion text.

## Installation
```bash
git clone https://github.com/freelawproject/blackletter
cd blackletter
pip install -e .
```

## Quick Start

**Command line:**
```bash
blackletter path/to/opinion.pdf -o output/folder -p 737
```

**Python:**
```python
from pathlib import Path
from blackletter import BlackletterPipeline

pipeline = BlackletterPipeline()
redacted_pdf, redacted_opinions, masked_opinions = pipeline.process(
    path_to_file,
    output_folder=Path("output"),
    first_page=1,
    redact=True,
    mask=True,
)
```

Or use the convenience function:
```python
from pathlib import Path
from blackletter import redact_pdf

redacted_pdf, redacted_opinions, masked_opinions = redact_pdf(path_to_file)
```

## Splitting Advance Sheets

An "advance sheet" is a legal reporter volume containing multiple judicial opinions. This tool can automatically split an advance sheet into individual opinion PDFs, identifying volume, reporter, and page ranges either through the Gemini API or manual metadata.

### Example: Split with known Metadata

If you already have metadata about the advance sheet (volume number, reporter, page ranges), provide it directly to skip the API call:
```python
from pathlib import Path
from blackletter import BlackletterPipeline, scan_splitter

pipeline = BlackletterPipeline()
raw_scan = Path("/filepath/to/advance_sheet.pdf")

# Manual metadata (skips LLM API)
metadata = [
    {
        "volume": 536,
        "reporter": "P.3d",
        "first_page": 737,
        "last_page": 1213,
    }
]

# Identify opinion boundaries and separate by volume/reporter
extracted_filepaths = scan_splitter(
    target_file=raw_scan,
    output_dir=Path("./output"),
    metadata=metadata,
)

# Process each opinion into redacted versions
for filepath in extracted_filepaths:
    opinions_filepath = Path(result['opinion_pdf'])
    parts = filepath.parts

    document = pipeline.process(
        pdf_path=filepath,
        first_page=int(parts[-2]),
        redact=True,
        mask=True,
        reduce=True,
        combine_short=True
    )
```

### Example: Split with Gemini API

If you don't have metadata, the tool can extract it automatically using the Gemini API: use `LLM_API_KEY`
```python
from pathlib import Path
from blackletter import BlackletterPipeline, scan_splitter
import os

pipeline = BlackletterPipeline()
raw_scan = Path("/filepath/to/advance_sheet.pdf")

# Set your Gemini API key
os.environ["LLM_API_KEY"] = "your-api-key"

# scan_splitter will automatically call gemini to extract metadata
extracted_filepaths = scan_splitter(
    target_file=raw_scan,
    output_dir=Path("./output"),
    # metadata parameter is optional - omit it to use Gemini
)

# Process results as above...
for filepath in extracted_filepaths:
    redacted_pdf, _, _ = pipeline.process(
        pdf_path=filepath,
        redact=True,
        mask=True,
        reduce=True,
    )
```

## How It Works

### Single Opinion Processing

The pipeline operates in four phases:

1. **Scanning (Phase 1)**: Uses YOLO to detect copyrighted elements (captions, key cites, headnotes, etc.)
2. **Planning (Phase 2)**: State machine determines which text spans to redact
3. **Execution (Phase 3)**: Applies redactions and masks to the PDF
4. **Extraction (Phase 4)**: Splits opinions into individual files (redacted and masked versions)

### Advance Sheet Splitting

For raw book scan splitting:

1. **Metadata Extraction**: Gemini API (or manual metadata) identifies opinion boundaries, volume numbers, and reporter information
2. **Section Detection**: YOLO finds the start of the opinion section (OPINION header)
3. **Job Planning**: Maps metadata to physical page locations and plans extraction jobs
4. **PDF Splitting**: Extracts individual opinions based on page ranges
5. **Individual Processing**: Each extracted opinion is processed through the single opinion pipeline (see above)

## Command Line Options
```bash
blackletter PDF [OPTIONS]

Positional Arguments:
  pdf                   Path to PDF file

Optional Arguments:
  -o, --output PATH         Output folder (default: pdf_parent/redactions)
  -v, --volume STR          The volume to redact
  -r, --reporter STR        The reporter to extract out
  -p, --page INT            First page number for case naming (default: 1)
  -m, --model PATH          Path to YOLO model (default: best.pt)
  -c, --confidence FLOAT    Confidence threshold (default: 0.20)
  -d, --dpi INT             DPI for PDF rendering (default: 200)
  --redact                  Generate unique redacted PDFs
  --mask                    Generate unique masked opinions
  --reduce                  Remove fully redacted pages from output
  --combine BOOL            Combine short opinion into single PDFs
  --combine-threshold INT   Threshold for combining short opinions
```

## Output

The pipeline produces three types of outputs:

1. **Redacted PDF**: Original PDF with copyrighted content marked for redaction
2. **Redacted Opinions**: Individual opinion PDFs extracted from the redacted document
3. **Masked Opinions**: Individual opinion PDFs with non-opinion content masked out

When processing advance sheets, each identified opinion is extracted and processed separately, producing all three output types for each opinion.

### Folders

Blackletter will generate the following folder structure and outputs

    /reporter/volume/page/opinion.pdf
    /reporter/volume/page/redactions/opinion_redacted.pdf
    /reporter/volume/page/redactions/masked/ [SEPARATE MASKED OPINIONS]
    /reporter/volume/page/redactions/redacted/ [SEPARATE REDACTED OPINIONS]


## Requirements

- Python 3.9+
- Gemini API key (for automatic metadata extraction; optional if using manual metadata)

## License

GNU Affero General Public License v3

## Contributing

Contributions welcome!