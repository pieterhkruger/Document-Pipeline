"""
Azure Document Intelligence integration for Document Pipeline.

This module provides OCR extraction using Azure Document Intelligence API
for scanned/non-OCRed documents.

Requirements:
    - Azure subscription
    - Document Intelligence resource
    - API key and endpoint (via environment variables)
    - Internet connectivity

Environment Variables:
    AZURE_DI_ENDPOINT: Azure DI endpoint URL
    AZURE_DI_KEY: Azure DI API key

Usage:
    from azure_document_intelligence import extract_ocr_with_azure_di

    result = extract_ocr_with_azure_di(
        document_path="path/to/document.pdf",
        verbose=True
    )

    if result:
        normalized_payload = result['normalized_payload']
        metadata = result['metadata']
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError
    AZURE_DI_AVAILABLE = True
except ImportError:
    AZURE_DI_AVAILABLE = False


def _convert_azure_result_to_dict(result: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of Azure DI SDK result objects into plain dictionaries."""
    if result is None:
        return None

    conversion_methods: List[str] = ["to_dict", "as_dict", "model_dump"]
    for method_name in conversion_methods:
        method: Optional[Callable] = getattr(result, method_name, None)
        if callable(method):
            try:
                converted = method()
                if isinstance(converted, dict):
                    return converted
            except TypeError:
                continue
            except Exception:
                continue

    to_generated = getattr(result, "_to_generated", None)
    if callable(to_generated):
        try:
            generated = to_generated()
            if generated is not None:
                generated_dict_method: Optional[Callable] = getattr(generated, "to_dict", None)
                if callable(generated_dict_method):
                    converted = generated_dict_method()
                    if isinstance(converted, dict):
                        return converted
        except Exception:
            pass

    return None


@dataclass
class AzureDIConfig:
    """Configuration for Azure Document Intelligence."""
    endpoint: str
    api_key: str
    model_id: str = "prebuilt-read"  # Use prebuilt-read for OCR

    @classmethod
    def from_environment(cls) -> Optional['AzureDIConfig']:
        """Load configuration from environment variables."""
        endpoint = os.getenv('AZURE_DI_ENDPOINT')
        api_key = os.getenv('AZURE_DI_KEY')

        if not endpoint or not api_key:
            return None

        return cls(endpoint=endpoint, api_key=api_key)


def is_azure_di_available() -> bool:
    """
    Check if Azure Document Intelligence is available.

    Returns:
        True if SDK installed and credentials configured, False otherwise
    """
    if not AZURE_DI_AVAILABLE:
        return False

    config = AzureDIConfig.from_environment()
    return config is not None


def extract_ocr_with_azure_di(
    document_path: str,
    verbose: bool = False,
    timeout: int = 120
) -> Optional[Dict[str, Any]]:
    """
    Extract OCR data from document using Azure Document Intelligence.

    Args:
        document_path: Path to PDF or image file
        verbose: Print progress messages
        timeout: Maximum wait time for processing (seconds)

    Returns:
        Dictionary with:
            - normalized_payload: Standard payload format (compatible with Docling)
            - raw_result: Original Azure DI result
            - metadata: Processing metadata (timing, page count, etc.)

        Returns None if extraction fails or credentials not available.
    """
    if not AZURE_DI_AVAILABLE:
        if verbose:
            print("[ERROR] Azure Document Intelligence SDK not installed")
            print("  Install: pip install azure-ai-documentintelligence")
        return None

    # Load configuration
    config = AzureDIConfig.from_environment()
    if not config:
        if verbose:
            print("[ERROR] Azure DI credentials not found")
            print("  Set environment variables: AZURE_DI_ENDPOINT, AZURE_DI_KEY")
        return None

    doc_path = Path(document_path)
    if not doc_path.exists():
        if verbose:
            print(f"[ERROR] Document not found: {document_path}")
        return None

    try:
        # Initialize client
        if verbose:
            print(f"[AZURE DI] Initializing client...")

        client = DocumentIntelligenceClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key)
        )

        # Start analysis
        start_time = time.time()

        if verbose:
            print(f"[AZURE DI] Reading document: {doc_path.name}")
            print(f"[AZURE DI] Starting OCR analysis (model: {config.model_id})...")

        # Determine content type based on file extension
        content_type_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        content_type = content_type_map.get(doc_path.suffix.lower(), 'application/octet-stream')

        # Open file and analyze (keep file open during API call)
        with open(doc_path, "rb") as f:
            poller = client.begin_analyze_document(
                config.model_id,  # First positional argument
                body=f,  # File object directly
                content_type=content_type
            )

        # Wait for completion
        if verbose:
            print(f"[AZURE DI] Waiting for results (timeout: {timeout}s)...")

        result = poller.result(timeout=timeout)

        elapsed = time.time() - start_time

        if verbose:
            print(f"[AZURE DI] Analysis complete ({elapsed:.1f}s)")
            print(f"  Pages: {len(result.pages)}")
            print(f"  Languages: {[lang.locale for lang in (result.languages or [])]}")

        # Normalize to standard format
        normalized_payload = _normalize_azure_di_result(result, verbose=verbose)

        # Extract metadata
        metadata = {
            'source': 'azure_document_intelligence',
            'model_id': config.model_id,
            'page_count': len(result.pages),
            'processing_time': elapsed,
            'languages': [lang.locale for lang in (result.languages or [])],
            'document_path': str(doc_path)
        }

        if verbose:
            print(f"[AZURE DI] Normalized payload generated")
            print(f"  Total pages: {len(normalized_payload)}")
            total_cells = sum(len(page.get('cells', [])) for page in normalized_payload)
            print(f"  Total cells: {total_cells}")

        raw_result_dict = _convert_azure_result_to_dict(result)
        payload: Dict[str, Any] = {
            'normalized_payload': normalized_payload,
            'metadata': metadata
        }

        if raw_result_dict is not None:
            payload['raw_result'] = raw_result_dict
        else:
            payload['raw_result'] = result
            payload['raw_result_object'] = result

        return payload

    except AzureError as e:
        if verbose:
            print(f"[ERROR] Azure DI API error: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"[ERROR] Unexpected error: {e}")
        return None


def _normalize_azure_di_result(result, verbose: bool = False) -> List[Dict]:
    """
    Normalize Azure DI result to standard payload format (Docling-compatible).

    Standard format:
    [
        {
            'page_num': 0,
            'width': 612.0,
            'height': 792.0,
            'cells': [
                {
                    'text': 'Sample text',
                    'bbox': {'x0': 10, 'y0': 20, 'x1': 100, 'y1': 40},
                    'confidence': 0.95,
                    'font': 'Unknown',  # Azure DI doesn't provide font info
                    'cell_id': 0
                },
                ...
            ]
        },
        ...
    ]
    """
    normalized_pages = []

    for page_idx, page in enumerate(result.pages):
        # Get page dimensions (convert from inches to points if needed)
        # Azure DI returns dimensions in inches, convert to points (1 inch = 72 points)
        width = page.width * 72 if page.unit == "inch" else page.width
        height = page.height * 72 if page.unit == "inch" else page.height

        # Extract words/lines into cells
        cells = []
        cell_id = 0

        # Process lines (each line becomes a cell)
        for line in (page.lines or []):
            # Get bounding box
            # Azure DI uses polygon format, convert to bbox (x0, y0, x1, y1)
            if line.polygon:
                # Polygon: [x1, y1, x2, y2, x3, y3, x4, y4]
                # Convert to bbox
                x_coords = [line.polygon[i] for i in range(0, len(line.polygon), 2)]
                y_coords = [line.polygon[i] for i in range(1, len(line.polygon), 2)]

                # Convert from inches to points if needed
                if page.unit == "inch":
                    x_coords = [x * 72 for x in x_coords]
                    y_coords = [y * 72 for y in y_coords]

                bbox = {
                    'x0': min(x_coords),
                    'y0': min(y_coords),
                    'x1': max(x_coords),
                    'y1': max(y_coords)
                }
            else:
                # No bbox available, skip
                continue

            # Get confidence (average of word confidences)
            confidence = 0.0
            if hasattr(line, 'words') and line.words:
                confidences = [w.confidence for w in line.words if hasattr(w, 'confidence') and w.confidence is not None]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0

            cell = {
                'text': line.content,
                'bbox': bbox,
                'confidence': confidence,
                'font': 'Unknown',  # Azure DI doesn't provide font information
                'cell_id': cell_id
            }

            cells.append(cell)
            cell_id += 1

        page_data = {
            'page_num': page_idx,
            'width': width,
            'height': height,
            'cells': cells
        }

        normalized_pages.append(page_data)

        if verbose and page_idx == 0:
            print(f"  [NORMALIZE] Page 0: {len(cells)} cells extracted")
            if cells:
                print(f"  [NORMALIZE] Sample cell: '{cells[0]['text'][:50]}...'")

    return normalized_pages


def test_azure_di_credentials() -> bool:
    """
    Test if Azure DI credentials are valid.

    Returns:
        True if credentials work, False otherwise
    """
    if not AZURE_DI_AVAILABLE:
        print("[TEST] Azure Document Intelligence SDK not installed")
        return False

    config = AzureDIConfig.from_environment()
    if not config:
        print("[TEST] Azure DI credentials not found in environment")
        return False

    try:
        # Try to initialize client
        client = DocumentIntelligenceClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key)
        )

        print(f"[TEST] Successfully initialized Azure DI client")
        print(f"  Endpoint: {config.endpoint}")
        print(f"  Model: {config.model_id}")
        return True

    except Exception as e:
        print(f"[TEST] Failed to initialize Azure DI client: {e}")
        return False
