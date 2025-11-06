# PDF OCR Detector

A standalone Streamlit application for detecting OCR status and security restrictions in PDF documents, with integrated Docling payload generation.

## Features

- **OCR Detection**: Determines whether a PDF contains searchable text (has been OCRed) or is a scanned document
- **Restriction Analysis**: Checks for PDF encryption and security restrictions including:
  - Printing permissions
  - Text copying/extraction permissions
  - Content modification permissions
  - Annotation permissions
- **Docling Payload Generation**: Generate enhanced Docling payloads with word-level and font information
  - Automatically saved to `Docling payloads/` directory
  - Configured with OCR, table structure, and parsed pages
  - Includes word-level cell data with:
    - Word bounding boxes
    - Font information (name, key)
    - Color information (RGBA)
    - OCR confidence scores
  - Shows payload metadata (size, pages, cells, tables)
  - One-click regeneration option
- **Font Analysis** (for enhanced payloads):
  - Font usage statistics showing count and percentage for each font
  - Interactive selectbox to filter text by font name (single font selection)
  - View all text items for the selected font, grouped by page
  - Displays metadata (font key, confidence, OCR flag) for each item
- **Hidden Text Detection** (for enhanced payloads):
  - Detects text hidden below other visible text (potential forgery indicator)
  - Creates flattened PDF (PDF → Images → PDF) to capture only visible text
  - Compares original vs flattened payloads to find hidden layers
  - Reports hidden text items with fonts and overlap metrics
  - Uses IoU (Intersection over Union) algorithm from Forgery_detection module
- **File Selection**: Easy-to-use selectbox interface to choose from test sample PDFs
- **Detailed Analysis**: Provides comprehensive information including:
  - Number of pages with text
  - Text preview from the first page
  - File size and metadata
  - Security and encryption status

## Installation

Make sure you have the required dependencies installed:

```bash
pip install streamlit pymupdf pikepdf docling pdf2image img2pdf
```

All dependencies are already included in the project's `requirements-313.txt` file.

**Required packages:**
- `streamlit` - Web interface framework
- `pymupdf` - PDF text extraction and basic analysis
- `pikepdf` - Detailed PDF permission checking (optional but recommended)
- `docling` - Raw payload generation for OCRed documents
- `pdf2image` - PDF to image conversion (required for hidden text detection)
- `img2pdf` - Image to PDF conversion (required for hidden text detection)

**Note:** `pdf2image` requires Poppler to be installed on your system:
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
- Add Poppler's `bin/` directory to your PATH

## Usage

1. Navigate to the Document Pipeline directory:
   ```bash
   cd "Document Pipeline"
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run pdf_ocr_detector.py
   ```

3. The application will open in your default web browser (typically at http://localhost:8501 or http://localhost:8502)

4. Use the selectbox to choose a PDF file from the test samples directory

5. View the analysis results:
   - **Left column**: OCR status and text content information
   - **Right column**: Security restrictions and permissions
   - **Bottom section**: Additional file information

## Test Samples

The application automatically loads PDF files from:
```
..\Test samples\Actual Use Cases
```

This directory contains various PDF documents for testing, including both legitimate and potentially fraudulent documents.

## How It Works

### OCR Detection
The application uses PyMuPDF (fitz) to:
1. Open the PDF document
2. Extract text from each page
3. Count pages with searchable text
4. Determine if the document has been OCRed

A PDF is considered "OCRed" if it contains any searchable text. Scanned documents without OCR will have no extractable text.

### Restriction Detection
The application uses two libraries for comprehensive restriction checking:

1. **pikepdf (preferred)**: Provides detailed permission flags for:
   - High/low-resolution printing
   - Text extraction
   - Content modification
   - Form field modification
   - Annotation editing

2. **PyMuPDF (fallback)**: Used when pikepdf is not available or fails
   - Checks encryption status
   - Detects password protection

### Docling Payload Generation
For documents with searchable text (OCRed), the application generates enhanced Docling payloads with word-level information:

1. **Configuration**: The DocumentConverter is configured with:
   - `do_ocr=True` - Enable OCR processing
   - `do_table_structure=True` - Detect and parse table structures
   - `generate_parsed_pages=True` - **KEY**: Enables cell-level word data extraction

2. **Processing**: The PDF is converted using Docling's pipeline

3. **Enhanced Export**: Creates a comprehensive payload with two components:
   - **docling_document**: Standard document structure (texts, tables, pages)
   - **parsed_pages**: Cell-level metadata extracted from `result.pages[].cells`
     - Each cell contains: text, bounding box (rect), font_name, font_key, rgba (color), confidence, from_ocr flag

4. **Storage**: Payloads are saved to `Document Pipeline/Docling payloads/` with naming convention:
   - `{document_name}_Docling_raw_payload.json`

5. **Reuse**: Existing payloads are detected and can be regenerated on demand

**Payload Structure:**
```json
{
  "docling_document": { ... },  // Standard Docling export
  "parsed_pages": {
    "0": {
      "page_no": 0,
      "size": {"width": 612, "height": 792},
      "cells": [
        {
          "text": "word",
          "rect": {"x0": ..., "y0": ..., "x1": ..., "y1": ...},
          "font_name": "Arial",
          "font_key": "F1",
          "rgba": [0, 0, 0, 255],
          "confidence": 0.99,
          "from_ocr": false
        }
      ]
    }
  },
  "metadata": { ... }
}
```

### Font Analysis
Once an enhanced payload exists, the application provides comprehensive font analysis:

1. **Font Statistics**: Analyzes all cells in the payload to generate:
   - Count of text items per font
   - Percentage distribution of fonts
   - Total unique fonts in document
   - Sorted by usage (most common first)

2. **Interactive Filtering**: Multiselect dropdown allows you to:
   - Select one or multiple fonts to analyze
   - View all text items using the selected font(s)
   - Text grouped by page for better organization

3. **Text Display**: For each selected font, shows:
   - All text items grouped by page
   - Font key identifier
   - OCR confidence score (if available)
   - Whether text came from OCR or native PDF
   - Number of items per page

This is useful for:
- Identifying font inconsistencies in documents
- Detecting potential forgeries (unusual font usage)
- Analyzing document structure and formatting
- Finding specific text by font type

### Hidden Text Detection
The application can detect text that is hidden below other visible text using dual-payload comparison:

1. **Flattening Process**:
   - Converts PDF to high-resolution images (300 DPI)
   - Converts images back to PDF
   - Result: PDF with only visible text (hidden layers removed)

2. **Payload Comparison**:
   - Original payload contains ALL text (visible + hidden)
   - Flattened payload contains only VISIBLE text
   - Compares word cells from both payloads

3. **Overlap Detection**:
   - Finds overlapping text cells using IoU (Intersection over Union)
   - IoU threshold: 25% (configurable via `IOU_THRESHOLD`)
   - Checks if both overlapping cells exist in visible payload
   - If one cell is missing → identified as hidden text

4. **Reporting**:
   - Shows hidden text and the visible text covering it
   - Displays fonts used for each layer
   - Reports overlap percentage (IoU)
   - Confidence level (HIGH for clear cases)

**Use Cases:**
- Detecting forged documents with hidden text layers
- Identifying tampering attempts
- Document authentication
- Forensic analysis of suspicious PDFs

**Detection Parameters:**
- `IOU_THRESHOLD = 0.25` - Minimum 25% overlap to consider
- `BBOX_TOLERANCE = 0.20` - 20% tolerance for bbox matching
- `DPI = 300` - Resolution for PDF flattening

## Output Information

For each PDF, the application displays:

### OCR Status
- Whether the PDF has been OCRed
- Total number of pages
- Number of pages containing text
- Percentage of pages with text
- Preview of text from the first page (expandable)

### Security Restrictions
- Encryption status
- Password protection status
- Detailed permissions (if available):
  - Printing allowed/restricted
  - Text copying allowed/restricted
  - Content modification allowed/restricted
  - Annotations allowed/restricted

### File Information
- File size (in MB)
- Full file path
- PDF version

### Docling Payload (OCRed documents only)
- **Enhanced payload indicator**: Shows if payload includes word-level detail & font info
- Payload existence status
- Payload file size
- Number of pages in payload
- **Number of word cells**: Total cells with individual word data
- Number of tables detected
- Last modification time
- **Font & color information**: Confirmation of included metadata
- Quick access to payloads folder

### Font Analysis (Enhanced payloads only)
- **Font statistics**: Total unique fonts and text items
- **Font usage table**: Each font with count and percentage
- **Interactive filter**: Selectbox to choose a font to display (single selection)
- **Filtered text display**: All text items for the selected font
  - Grouped by page
  - Shows font key, confidence, OCR flag
  - Scrollable text areas per page

### Hidden Text Detection (Enhanced payloads only)
- **Detection button**: One-click hidden text analysis
- **Flattening info**: Shows the flattening process explanation
- **Detection results**:
  - Count of hidden text items found
  - Total overlaps checked
  - Page-by-page breakdown
- **Detailed view for each hidden item**:
  - Hidden text content and font
  - Visible text (covering layer) and font
  - Overlap percentage (IoU)
  - Confidence level
- **Generated files**: Lists flattened PDF and visible payload files

## Technical Details

- **Primary PDF Library**: PyMuPDF (fitz) v1.24.4+
- **Security Analysis Library**: pikepdf (optional but recommended)
- **Payload Generation**: Docling v2.52.0+
- **Frontend Framework**: Streamlit v1.38+
- **Python Version**: 3.13+ (compatible with earlier versions)

## File Structure

```
Document Pipeline/
├── pdf_ocr_detector.py       # Main application
├── README.md                  # This file
└── Docling payloads/          # Generated payloads (auto-created)
    └── {document_name}_Docling_raw_payload.json
```

## Notes

- The application is read-only and does not modify any PDF files
- For password-protected PDFs, the application will detect the encryption but cannot analyze content without the password
- Some PDFs may have encryption but still be openable (encryption without password protection)
- Docling payload generation only appears for documents with searchable text (OCRed documents)
- **Enhanced payloads** include word-level cell data with font and color information
- Payloads are reused if they already exist (use "Regenerate" to override)
- If you have old payloads without word-level data, regenerate them to get the enhanced format
- All Docling processing is done locally - no external API calls
- Cell data structure follows the pattern used in Forgery_detection for compatibility
