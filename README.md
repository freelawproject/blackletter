# Blackletter

Remove copyrighted material from legal case law PDFs.

A reference to blackletter law, this tool removes proprietary annotations from judicial opinions—specifically, headnotes, and other copyrighted materials—while preserving the authentic opinion text.

## Installation

```bash
git clone https://github.com/yourusername/blackletter
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

Another example, that takes the raw scan and isolates the opinion context before splitting 
```
from pathlib import Path
from blackletter import BlackletterPipeline, scan_splitter

pipeline = BlackletterPipeline()
raw_scan = "/filepath/to/scan.pdf"

# you can give it manual metadata if you dont want to use the LLM
metadata = [{"id": 0, "volume": 536, "reporter": "P.3d", "pages": {"start": 737, "end": 1213}}]

OR 

use an env variable LLM_API_KEY for gemini

# Identify the opinion content and separate into volume reporter pages
results = scan_splitter(
    target_file=Path(raw_scan),
    model=pipeline.model,
    output_dir="/output/directory/here/",
    metadata=metadata,
)

# Process each individual volume into its redacted opinions
for result in results:
    opinions_filepath = Path(result['opinion_pdf'])
    parts = Path(opinions_filepath).parts 
    redacted_pdf, _, _ = pipeline.process(
        pdf_path=filepath,
        first_page=int(parts[-2]),
        redact=True,
        mask=True,
        reduce=True,
    )

```


## How It Works

The pipeline operates in four phases:

1. **Scanning (Phase 1)**: Uses YOLO to detect copyrighted elements (captions, key cites, headnotes, etc.)
2. **Planning (Phase 2)**: State machine determines which text spans to redact
3. **Execution (Phase 3)**: Applies redactions and masks to the PDF
4. **Extraction (Phase 4)**: Splits opinions into individual files (redacted and masked versions)

## Command Line Options

```bash
blackletter PDF [OPTIONS]

Positional Arguments:
  pdf                   Path to PDF file

Optional Arguments:
  -o, --output PATH     Output folder (default: pdf_parent/redactions)
  -v, --volume STR      The volume to redact
  -r, --reporter STR    The reporter to extract out
  -p, --page INT        First page number for case naming (default: 1)
  -m, --model PATH      Path to YOLO model (default: best.pt)
  -c, --confidence FLOAT Confidence threshold (default: 0.20)
  -d, --dpi INT         DPI for PDF rendering (default: 200)
  --redact              Generate unique redacted PDFs
  --mask                Generate unique masked opinions
  --reduce              Remove fully redacted pages from masked output
```

## Configuration

Configure behavior via the RedactionConfig object:

```python
from blackletter import BlackletterPipeline
from blackletter.config import RedactionConfig

config = RedactionConfig(
    MODEL_PATH="best.pt",
    confidence_threshold=0.20,
    dpi=200,
)

pipeline = BlackletterPipeline(config)
redacted_pdf, redacted_opinions, masked_opinions = pipeline.process("opinion.pdf")
```

## Output

The pipeline produces three outputs:

1. **Redacted PDF**: Original PDF with copyrighted content marked for redaction
2. **Redacted Opinions**: Individual opinion PDFs extracted from the redacted document
3. **Masked Opinions**: Individual opinion PDFs with non-opinion content masked out

## Requirements

- Python 3.8+
- YOLO model (`best.pt`)
- PDFs must be text-based (not scanned images)

## License

MIT

## Contributing

Contributions welcome! Please submit issues and PRs.