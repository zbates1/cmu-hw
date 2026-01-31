# PDF Batch Extractor Skill

You are an expert at extracting text from PDF documents, especially large ones that need to be processed in batches. Your role is to read PDFs page by page or in configurable batch sizes and return extracted text.

## How to Use

When the user provides a PDF file path:
1. First, read the PDF to determine total page count
2. If the PDF is small (under 20 pages), extract all text at once
3. If the PDF is large, process in batches of configurable size (default: 10 pages per batch)
4. Report progress after each batch
5. Combine results or process each batch as requested

## Batch Processing Strategy

### Determining Batch Size
- **Small PDFs (1-20 pages)**: Process all at once
- **Medium PDFs (21-100 pages)**: 10-20 pages per batch
- **Large PDFs (100+ pages)**: 20-50 pages per batch
- Adjust based on page content density (image-heavy pages need smaller batches)

### Processing Flow
```
1. Open PDF and get total page count
2. Calculate number of batches needed
3. For each batch:
   a. Extract text from pages in batch
   b. Report batch progress (e.g., "Processed pages 1-10 of 150")
   c. Store/return batch results
4. Combine all batches if full text requested
```

## Implementation

Use the Read tool to read PDFs directly - Claude Code can process PDF files and extract text and visual content page by page.

### For Programmatic Extraction (Python)

When the user needs a script for batch PDF extraction:

```python
import fitz  # PyMuPDF
from pathlib import Path
from typing import Generator, Optional

def extract_pdf_text_batched(
    pdf_path: str,
    batch_size: int = 10,
    start_page: int = 0,
    end_page: Optional[int] = None
) -> Generator[dict, None, None]:
    """
    Extract text from PDF in batches.

    Args:
        pdf_path: Path to PDF file
        batch_size: Number of pages per batch
        start_page: First page to process (0-indexed)
        end_page: Last page to process (exclusive), None for all pages

    Yields:
        dict with keys: batch_num, start_page, end_page, text, page_count
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    end_page = end_page or total_pages

    batch_num = 0
    for batch_start in range(start_page, end_page, batch_size):
        batch_end = min(batch_start + batch_size, end_page)
        batch_text = []

        for page_num in range(batch_start, batch_end):
            page = doc[page_num]
            text = page.get_text()
            batch_text.append(f"--- Page {page_num + 1} ---\n{text}")

        yield {
            "batch_num": batch_num,
            "start_page": batch_start + 1,  # 1-indexed for display
            "end_page": batch_end,
            "text": "\n\n".join(batch_text),
            "total_pages": total_pages
        }
        batch_num += 1

    doc.close()


def extract_all_text(pdf_path: str, batch_size: int = 10) -> str:
    """Extract all text from PDF, processing in batches internally."""
    all_text = []
    for batch in extract_pdf_text_batched(pdf_path, batch_size):
        all_text.append(batch["text"])
        print(f"Processed pages {batch['start_page']}-{batch['end_page']} of {batch['total_pages']}")
    return "\n\n".join(all_text)


def save_batches_to_files(
    pdf_path: str,
    output_dir: str,
    batch_size: int = 10
) -> list[str]:
    """Save each batch to a separate text file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    output_files = []

    for batch in extract_pdf_text_batched(pdf_path, batch_size):
        filename = f"{pdf_name}_pages_{batch['start_page']}-{batch['end_page']}.txt"
        filepath = output_path / filename
        filepath.write_text(batch["text"], encoding="utf-8")
        output_files.append(str(filepath))
        print(f"Saved: {filename}")

    return output_files
```

### Alternative: Using pdfplumber

```python
import pdfplumber
from typing import Generator, Optional

def extract_with_pdfplumber(
    pdf_path: str,
    batch_size: int = 10
) -> Generator[dict, None, None]:
    """Extract text using pdfplumber (better for tables)."""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_text = []

            for page_num in range(batch_start, batch_end):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                batch_text.append(f"--- Page {page_num + 1} ---\n{text}")

            yield {
                "start_page": batch_start + 1,
                "end_page": batch_end,
                "text": "\n\n".join(batch_text),
                "total_pages": total_pages
            }
```

## Dependencies

Install required libraries:
```bash
pip install pymupdf  # For fitz/PyMuPDF
# OR
pip install pdfplumber  # Alternative with better table support
```

## Usage Examples

### Direct Extraction via Claude
```
User: Extract text from report.pdf
Assistant: [Uses Read tool to read the PDF, reports content batch by batch if large]
```

### Script-Based Extraction
```python
# Process large PDF in batches
for batch in extract_pdf_text_batched("large_document.pdf", batch_size=20):
    print(f"Batch {batch['batch_num']}: pages {batch['start_page']}-{batch['end_page']}")
    # Process batch['text'] as needed

# Save to separate files
files = save_batches_to_files("textbook.pdf", "./extracted/", batch_size=25)

# Get all text with progress reporting
full_text = extract_all_text("manual.pdf")
```

## Handling Special Cases

### Image-Heavy PDFs
- Use OCR for scanned documents (add `pytesseract` integration)
- Reduce batch size for memory efficiency

### PDFs with Tables
- Use `pdfplumber` for better table extraction
- Consider `tabula-py` for structured table data

### Encrypted PDFs
```python
doc = fitz.open(pdf_path)
if doc.is_encrypted:
    doc.authenticate(password)  # User must provide password
```

### Memory Optimization
- Process and discard each batch immediately if only counting/searching
- Use generators to avoid loading all text into memory
- For very large PDFs, save batches to disk

## Output Options

1. **Combined Text**: Single string with all content
2. **Batch Generator**: Yield batches for streaming processing
3. **Separate Files**: Save each batch to individual text files
4. **Structured Data**: Return dict with page numbers and metadata

## Best Practices

- Always close PDF documents after processing
- Report progress for user feedback on large files
- Handle encoding issues gracefully (use `errors='ignore'` if needed)
- Validate PDF exists and is readable before processing
- Consider memory limits when setting batch sizes
