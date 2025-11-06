"""
PDF OCR Detection Application
==============================
Streamlit application to detect OCR status and restrictions in PDF documents,
and generate enhanced Docling payloads with word-level and font information.

This tool analyzes PDFs to determine:
1. Whether the PDF has been OCRed (contains searchable text)
2. Whether the PDF has security restrictions
3. Generate enhanced Docling payloads with:
   - Document structure (standard Docling export)
   - Word-level cell data with bounding boxes
   - Font information (name, key) for each word
   - Color information (RGBA) for each word
   - OCR confidence scores

Payloads are saved to: Docling payloads/

Usage:
    streamlit run pdf_ocr_detector.py
"""

import streamlit as st
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict
import math
import os
import json
import time

# Try to import pikepdf for better permission checking
try:
    import pikepdf
    HAS_PIKEPDF = True
except ImportError:
    HAS_PIKEPDF = False

# Try to import Docling for payload generation
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

# Try to import Azure DI
try:
    from azure_document_intelligence import extract_ocr_with_azure_di, is_azure_di_available
    HAS_AZURE_DI = is_azure_di_available()
except ImportError:
    HAS_AZURE_DI = False

# Try to import cluster analysis
try:
    from cluster_analysis import perform_cluster_analysis, get_human_explanation_for_feature
    import numpy as np
    HAS_CLUSTER_ANALYSIS = True
except ImportError:
    HAS_CLUSTER_ANALYSIS = False

TESSERACT_CMD_PATH: Optional[Path] = None
TESSDATA_DIR_PATH: Optional[Path] = None

# Try to import pdf2image and pytesseract for hidden text detection
try:
    from pdf2image import convert_from_path
    import pytesseract

    def _set_tessdata_env(tessdata_dir: Path) -> Path:
        """
        Normalize and apply Tesseract tessdata environment variables.
        Ensures the directory exists and updates TESSDATA_PREFIX with a trailing separator.
        """
        if not tessdata_dir or not tessdata_dir.exists():
            return tessdata_dir

        normalized = tessdata_dir
        tessdata_str = str(normalized)
        if not tessdata_str.endswith(os.sep) and not tessdata_str.endswith('/'):
            tessdata_str = tessdata_str + os.sep
        os.environ['TESSDATA_PREFIX'] = tessdata_str
        return normalized

    def _resolve_tesseract_installation() -> Tuple[Optional[Path], Optional[Path]]:
        """
        Locate the Tesseract executable and tessdata directory using environment variables
        and common installation paths.
        """
        candidate_cmds: List[Path] = []

        env_cmd = os.environ.get('TESSERACT_CMD') or os.environ.get('TESSERACT_PATH')
        if env_cmd:
            candidate_cmds.append(Path(env_cmd.strip('"')))

        if os.name == 'nt':
            program_files = Path(os.environ.get('PROGRAMFILES', r"C:\Program Files"))
            program_files_x86 = Path(os.environ.get('PROGRAMFILES(X86)', r"C:\Program Files (x86)"))
            potential_roots = [
                Path(r"C:\Program Files\Tesseract-OCR"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR"),
                program_files / "Tesseract-OCR",
                program_files_x86 / "Tesseract-OCR",
            ]
            for root in potential_roots:
                candidate_cmds.append(root / "tesseract.exe")
        else:
            candidate_cmds.extend(
                Path(p) for p in ("/usr/bin/tesseract", "/usr/local/bin/tesseract")
            )

        found_cmd: Optional[Path] = None
        for cmd_path in candidate_cmds:
            if cmd_path and cmd_path.exists():
                found_cmd = cmd_path
                break

        env_tessdata_candidates: List[Path] = []
        for env_key in ("TESSDATA_PREFIX", "TESSDATA_DIR", "TESSERACT_TESSDATA"):
            env_val = os.environ.get(env_key)
            if not env_val:
                continue
            candidate = Path(env_val.strip('"'))
            if candidate.exists():
                env_tessdata_candidates.append(candidate)

        found_tessdata: Optional[Path] = None
        for candidate in env_tessdata_candidates:
            if candidate.name.lower() == "tessdata" and candidate.is_dir():
                found_tessdata = candidate
                break
            maybe = candidate / "tessdata"
            if maybe.is_dir():
                found_tessdata = maybe
                break

        if not found_tessdata and found_cmd:
            default_candidate = found_cmd.parent / "tessdata"
            if default_candidate.is_dir():
                found_tessdata = default_candidate

        if not found_tessdata and os.name == 'nt':
            for fallback in (
                Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
            ):
                if fallback.is_dir():
                    found_tessdata = fallback
                    break

        return found_cmd, found_tessdata

    TESSERACT_CMD_PATH, TESSDATA_DIR_PATH = _resolve_tesseract_installation()
    if TESSERACT_CMD_PATH:
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD_PATH)

    if TESSDATA_DIR_PATH and TESSDATA_DIR_PATH.is_dir():
        TESSDATA_DIR_PATH = _set_tessdata_env(TESSDATA_DIR_PATH)
    else:
        existing_prefix = os.environ.get('TESSDATA_PREFIX')
        if existing_prefix:
            prefix_path = Path(existing_prefix.strip('"'))
            if prefix_path.is_dir():
                if prefix_path.name.lower() == "tessdata":
                    TESSDATA_DIR_PATH = _set_tessdata_env(prefix_path)
                else:
                    nested = prefix_path / "tessdata"
                    if nested.is_dir():
                        TESSDATA_DIR_PATH = _set_tessdata_env(nested)
            elif prefix_path.exists():
                nested = prefix_path / "tessdata"
                if nested.is_dir():
                    TESSDATA_DIR_PATH = _set_tessdata_env(nested)

    HAS_FLATTENING = True
except ImportError:
    HAS_FLATTENING = False
    TESSERACT_CMD_PATH = None
    TESSDATA_DIR_PATH = None

# Try to import PIL for image manipulation and drawing
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configuration
TEST_SAMPLES_PATH = Path(__file__).parent / "Actual Use Cases"
# UNIFIED RAW PAYLOAD LOCATION - All tools save raw API responses here
RAW_PAYLOADS_PATH = Path(__file__).parent.parent / "Glyph_anomalies" / "raw_payloads"
# Legacy locations (kept for backwards compatibility in reading)
PAYLOADS_PATH = Path(__file__).parent / "Docling payloads"
ADOBE_PAYLOADS_PATH = Path(__file__).parent / "Adobe payloads"
PIKEPDF_PAYLOADS_PATH = Path(__file__).parent / "pikepdf payloads"
ENHANCED_TEXT_INFO_PATH = Path(__file__).parent / "Enhanced text info"
AZURE_DI_PAYLOADS_PATH = Path(__file__).parent / "Azure DI payloads"
CLUSTER_RESULTS_PATH = Path(__file__).parent / "Cluster Results"
OVERLAPPING_BBOXES_PATH = Path(__file__).parent / "Overlapping bboxes"
PDF_PROCESSING_PATH = Path(__file__).parent / "PDF processing"
# Using APIs if available
USE_ADOBE_API_IF_AVAILABLE = True
USE_AZURE_DI_API_IF_AVAILABLE = False
USE_GOOGLE_API_IF_AVAILABLE = False


# Hidden text detection parameters
IOU_THRESHOLD = 0.10  # Minimum overlap to consider hidden text (10%)
BBOX_TOLERANCE = 0.20  # Tolerance for bbox matching in visible payload (20%)
DPI = 400  # DPI for PDF->Image conversion (hidden text detection only)
KEEP_INTERMEDIATE_FILES = False  # Set to True to keep temp images/PDFs

# Feature extraction DPI (color clustering & edge gradient analysis)
FEATURE_EXTRACTION_DPI = 1200  # Maximum DPI - actual DPI may be lower based on source images

# Alignment anomaly thresholds for Docling label/value analysis (points)
DOCLING_ALIGNMENT_SPACING_THRESHOLD = 3.0
DOCLING_ALIGNMENT_LEFT_THRESHOLD = 0.75
DOCLING_ALIGNMENT_VERTICAL_THRESHOLD = 1.0
DOCLING_ALIGNMENT_RIGHT_THRESHOLD = 1.5

# Colon pattern detection thresholds (for label:value spacing analysis)
COLON_SPACING_TOLERANCE = 2.0          # pt - deviation threshold for flagging anomalies
COLON_MAX_DISTANCE = 75.0              # pt - maximum label-to-value distance (increased from 50.0)
COLON_BASELINE_TOLERANCE = 3.0         # pt - vertical alignment tolerance (increased from 2.0)
COLON_MIN_CLUSTER_SIZE = 3             # minimum items to establish a pattern
COLON_RIGHT_ALIGN_TOLERANCE = 1.5      # pt - right edge alignment tolerance

# Flattening segmentation mismatch detection thresholds
SEGMENTATION_CELL_RATIO_THRESHOLD = 1.40   # Visible payload cells at least 40% greater than original
SEGMENTATION_AVG_WORD_RATIO_THRESHOLD = 0.75  # Visible average words per cell at most 75% of original

# Text matching thresholds
WORD_LEVEL_AVG_THRESHOLD = 1.30          # Avg words/cell <= 1.30 considered word-level
WORD_LEVEL_SINGLE_WORD_SHARE = 0.65      # >=65% single-word cells treated as word-level
WORD_SIMILARITY_THRESHOLD = 0.92         # Required similarity for word-level matches
PHRASE_SIMILARITY_THRESHOLD = 0.85       # Required similarity for phrase-level matches
MIN_CANDIDATE_IOU = 0.05                 # Minimum IoU to consider a visible candidate
LENIENT_WORD_SIMILARITY_THRESHOLD = 0.70
LENIENT_PHRASE_SIMILARITY_THRESHOLD = 0.60
SHOW_TEXT_NOT_RENDERED = False            # Toggle reporting/display of non-visible text items
SHOW_ANOMALY_DETECTION = False            # Toggle anomaly detection reporting sections

# Try to import adobe runner for payload generation
try:
    import sys
    import importlib.util

    # Add paths for importing adobe runner
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TOOLS_DIR = PROJECT_ROOT / "Unit testing - Tools" / "tools"

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Import adobe runner
    def _import_adobe_runner():
        runner_path = TOOLS_DIR / "adobe" / "runner.py"
        if not runner_path.exists():
            return None

        module_name = "tools.adobe.runner"
        spec = importlib.util.spec_from_file_location(module_name, runner_path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        module.__package__ = "tools.adobe"

        # Register in sys.modules
        sys.modules[module_name] = module

        # Create package module
        if "tools.adobe" not in sys.modules:
            import types
            pkg_module = types.ModuleType("tools.adobe")
            pkg_module.__path__ = [str(TOOLS_DIR / "adobe")]
            pkg_module.__package__ = "tools"
            sys.modules["tools.adobe"] = pkg_module

        spec.loader.exec_module(module)

        if not hasattr(module, 'run_tool'):
            return None

        return module.run_tool

    adobe_run_tool = _import_adobe_runner()
    HAS_ADOBE = adobe_run_tool is not None
except:
    HAS_ADOBE = False
    adobe_run_tool = None


def _check_env_variables(required_vars: list) -> bool:
    """
    Helper function to check if required environment variables are set in .env file.

    Args:
        required_vars: List of required environment variable names.
                      Can contain strings (exact match) or tuples (any one of alternatives)

    Returns:
        True if .env doesn't exist (allows system-wide env vars) OR
        True if .env exists and all required variables are set, False otherwise
    """
    env_file = Path(__file__).parent / ".env"

    # If .env doesn't exist, assume environment variables may be set system-wide
    if not env_file.exists():
        return True

    # If .env exists, validate that all required variables are present
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)

        for var in required_vars:
            if isinstance(var, (list, tuple)):
                # Any one of these alternatives must be set
                if not any(os.getenv(v) for v in var):
                    return False
            else:
                # This specific variable must be set
                if not os.getenv(var):
                    return False
        return True
    except ImportError:
        # If dotenv not available, environment variables may be set system-wide
        return True


def is_adobe_api_enabled() -> bool:
    """
    Check if Adobe API is enabled and available.

    Returns True if:
    1. USE_ADOBE_API_IF_AVAILABLE is True
    2. HAS_ADOBE is True (Adobe runner is available)
    3. Either .env doesn't exist OR .env exists with required Adobe API keys
    """
    if not USE_ADOBE_API_IF_AVAILABLE:
        return False

    if not HAS_ADOBE:
        return False

    # Check for Adobe API credentials in .env
    # Adobe typically uses client credentials or service account
    return _check_env_variables([
        'ADOBE_CLIENT_ID',
        'ADOBE_CLIENT_SECRET'
    ])


def is_azure_di_api_enabled() -> bool:
    """
    Check if Azure DI API is enabled and available.

    Returns True if:
    1. USE_AZURE_DI_API_IF_AVAILABLE is True
    2. HAS_AZURE_DI is True (Azure DI SDK is available)
    3. Either .env doesn't exist OR .env exists with AZURE_DI_ENDPOINT and AZURE_DI_KEY
    """
    if not USE_AZURE_DI_API_IF_AVAILABLE:
        return False

    if not HAS_AZURE_DI:
        return False

    return _check_env_variables([
        'AZURE_DI_ENDPOINT',
        'AZURE_DI_KEY'
    ])


def is_google_api_enabled() -> bool:
    """
    Check if Google Document AI API is enabled and available.

    Returns True if:
    1. USE_GOOGLE_API_IF_AVAILABLE is True
    2. Either .env doesn't exist OR .env exists with Google credentials
       (GOOGLE_APPLICATION_CREDENTIALS and either GCP_PROJECT_ID or GOOGLE_DOC_AI_PROJECT_ID)

    Note: We check configuration but don't import the Google module here
    """
    if not USE_GOOGLE_API_IF_AVAILABLE:
        return False

    return _check_env_variables([
        'GOOGLE_APPLICATION_CREDENTIALS',
        ('GCP_PROJECT_ID', 'GOOGLE_DOC_AI_PROJECT_ID'),
        ('DOC_AI_PROCESSOR_ID', 'GOOGLE_DOC_AI_PROCESSOR_ID')
    ])


def extract_font_properties_with_pikepdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract font properties using pikepdf when Adobe API is not available.

    Returns:
        Dictionary mapping font reference to properties:
        {
            font_ref: {
                'embedded': bool,
                'encoding': str,
                'font_type': str,
                'name': str
            }
        }
    """
    if not HAS_PIKEPDF:
        return {}

    try:
        pdf = pikepdf.open(pdf_path)
        font_properties = {}

        for page_num, page in enumerate(pdf.pages, start=1):
            resources = page.get("/Resources")
            if not resources or not isinstance(resources, pikepdf.Dictionary):
                continue

            fonts = resources.get("/Font")
            if not fonts or not isinstance(fonts, pikepdf.Dictionary):
                continue

            for font_name, font_ref in fonts.items():
                # Resolve indirect reference
                try:
                    font_obj = font_ref
                    if hasattr(font_ref, 'get_object'):
                        font_obj = font_ref.get_object()

                    if not isinstance(font_obj, pikepdf.Dictionary):
                        continue

                    # Create unique key for this font
                    font_key = str(font_name)

                    # Skip if already processed
                    if font_key in font_properties:
                        continue

                    # Extract font properties
                    props = {}

                    # 1. Check if font is embedded
                    # Look for /FontDescriptor → /FontFile, /FontFile2, or /FontFile3
                    font_descriptor = font_obj.get("/FontDescriptor")
                    if font_descriptor:
                        if hasattr(font_descriptor, 'get_object'):
                            font_descriptor = font_descriptor.get_object()

                        if isinstance(font_descriptor, pikepdf.Dictionary):
                            has_font_file = (
                                "/FontFile" in font_descriptor or
                                "/FontFile2" in font_descriptor or
                                "/FontFile3" in font_descriptor
                            )
                            props['embedded'] = has_font_file
                        else:
                            props['embedded'] = False
                    else:
                        props['embedded'] = False

                    # 2. Extract encoding
                    encoding = font_obj.get("/Encoding")
                    if encoding:
                        if isinstance(encoding, pikepdf.Name):
                            props['encoding'] = str(encoding)[1:]  # Remove leading '/'
                        elif isinstance(encoding, pikepdf.Dictionary):
                            # Complex encoding - check BaseEncoding
                            base_encoding = encoding.get("/BaseEncoding")
                            if base_encoding and isinstance(base_encoding, pikepdf.Name):
                                props['encoding'] = str(base_encoding)[1:]
                            else:
                                props['encoding'] = "CustomEncoding"
                        else:
                            props['encoding'] = str(encoding)
                    else:
                        # For Type0 fonts, check descendant font's CIDSystemInfo
                        subtype = font_obj.get("/Subtype")
                        if subtype == pikepdf.Name("/Type0"):
                            descendant_fonts = font_obj.get("/DescendantFonts")
                            if descendant_fonts and isinstance(descendant_fonts, pikepdf.Array):
                                if len(descendant_fonts) > 0:
                                    desc_font = descendant_fonts[0]
                                    if hasattr(desc_font, 'get_object'):
                                        desc_font = desc_font.get_object()

                                    if isinstance(desc_font, pikepdf.Dictionary):
                                        cid_info = desc_font.get("/CIDSystemInfo")
                                        if cid_info:
                                            if hasattr(cid_info, 'get_object'):
                                                cid_info = cid_info.get_object()

                                            if isinstance(cid_info, pikepdf.Dictionary):
                                                registry = cid_info.get("/Registry")
                                                ordering = cid_info.get("/Ordering")
                                                if registry and ordering:
                                                    props['encoding'] = f"{str(registry)}-{str(ordering)}"
                                                else:
                                                    props['encoding'] = "Identity-H"
                                            else:
                                                props['encoding'] = "Identity-H"
                                        else:
                                            props['encoding'] = "Identity-H"
                                    else:
                                        props['encoding'] = None
                                else:
                                    props['encoding'] = None
                            else:
                                props['encoding'] = None
                        else:
                            props['encoding'] = None

                    # 3. Extract font type
                    # Get /Subtype (e.g., /Type0, /Type1, /TrueType)
                    subtype = font_obj.get("/Subtype")
                    if subtype and isinstance(subtype, pikepdf.Name):
                        font_type = str(subtype)[1:]  # Remove leading '/'

                        # For Type0 fonts, also check descendant font type
                        if font_type == "Type0":
                            descendant_fonts = font_obj.get("/DescendantFonts")
                            if descendant_fonts and isinstance(descendant_fonts, pikepdf.Array):
                                if len(descendant_fonts) > 0:
                                    desc_font = descendant_fonts[0]
                                    if hasattr(desc_font, 'get_object'):
                                        desc_font = desc_font.get_object()

                                    if isinstance(desc_font, pikepdf.Dictionary):
                                        desc_subtype = desc_font.get("/Subtype")
                                        if desc_subtype and isinstance(desc_subtype, pikepdf.Name):
                                            desc_type = str(desc_subtype)[1:]
                                            font_type = f"{font_type}+{desc_type}"

                        props['font_type'] = font_type
                    else:
                        props['font_type'] = "Unknown"

                    # 4. Extract font name
                    base_font = font_obj.get("/BaseFont")
                    if base_font and isinstance(base_font, pikepdf.Name):
                        props['name'] = str(base_font)[1:]  # Remove leading '/'
                    else:
                        props['name'] = font_key

                    font_properties[font_key] = props

                except Exception as e:
                    # Skip problematic fonts
                    continue

        pdf.close()
        return font_properties

    except Exception as e:
        st.error(f"Error extracting font properties with pikepdf: {e}")
        return {}


# HasClip detection functions (adapted from detect_hasclip.py)
TEXT_SHOW_OPS = {"Tj", "TJ", "'", '"'}   # text painting ops
CLIP_OPS = {"W", "W*"}                   # clip path operators

if HAS_PIKEPDF:
    PIKEPDF_PARSE_CONTENT = getattr(pikepdf, "parsecontent", None)
    if PIKEPDF_PARSE_CONTENT is None:
        PIKEPDF_PARSE_CONTENT = getattr(pikepdf, "parse_content_stream", None)
else:
    PIKEPDF_PARSE_CONTENT = None


class GraphicsState:
    """
    Minimal graphics state to answer 'is there a clipping path active?'.
    Real PDFs have more, but this is enough to mimic HasClip.
    """
    __slots__ = ("clipped",)

    def __init__(self, clipped: bool = False):
        self.clipped = clipped

    def clone(self) -> "GraphicsState":
        return GraphicsState(self.clipped)


def _ensure_dictionary(obj: Any) -> Optional[pikepdf.Dictionary]:
    """
    Safely convert a pikepdf object to a Dictionary if possible.
    """
    if not HAS_PIKEPDF:
        return None

    if isinstance(obj, pikepdf.Dictionary):
        return obj

    try:
        return pikepdf.Dictionary(obj)
    except Exception:
        return None


def _get_xobject(resources: Any, name: pikepdf.Name) -> Optional[pikepdf.Stream]:
    """
    Resolve a Form XObject by name from the given resources dictionary.
    """
    if not resources:
        return None
    resources_dict = _ensure_dictionary(resources)
    if resources_dict is None:
        return None

    xobjs = resources_dict.get("/XObject", None)
    if not isinstance(xobjs, (pikepdf.Dictionary, pikepdf.Object)):
        return None

    if isinstance(xobjs, pikepdf.Object):
        try:
            xobjs = pikepdf.Dictionary(xobjs)
        except Exception:
            return None

    stream = xobjs.get(name, None)
    if stream is None:
        return None
    try:
        # Indirect objects need resolving
        return stream.get_object() if hasattr(stream, "get_object") else stream
    except Exception:
        return None


def _xobject_resources(xobj: pikepdf.Stream, inherited: Optional[pikepdf.Dictionary]) -> Optional[pikepdf.Dictionary]:
    """
    Determine which /Resources apply inside the XObject.
    Form XObjects may have their own /Resources; otherwise inherit.
    """
    own = xobj.get("/Resources", None)
    own_dict = _ensure_dictionary(own)
    if own_dict is not None:
        return own_dict
    return _ensure_dictionary(inherited)


def _iter_content_operations(content_stream: Any):
    """
    Yield (operands, operator) pairs for any content stream representation.
    """
    if not HAS_PIKEPDF or PIKEPDF_PARSE_CONTENT is None:
        return

    if content_stream is None:
        return

    if isinstance(content_stream, (pikepdf.Array, list, tuple)):
        for item in content_stream:
            yield from _iter_content_operations(item)
        return

    try:
        instructions = PIKEPDF_PARSE_CONTENT(content_stream)
    except Exception:
        if isinstance(content_stream, pikepdf.Object):
            try:
                resolved = content_stream.get_object()
            except Exception:
                resolved = None
            if resolved is not None and resolved is not content_stream:
                yield from _iter_content_operations(resolved)
        return

    for instruction in instructions:
        yield instruction.operands, instruction.operator


def extract_text_from_operands(op_name: str, operands: Any) -> str:
    """
    Best-effort extraction of literal text from Tj/TJ/'/" operands.
    """
    if not HAS_PIKEPDF:
        return ""

    try:
        if op_name == "Tj":
            return str(operands[0])
        if op_name == "TJ":
            parts = []
            for item in operands[0]:
                if isinstance(item, (str, pikepdf.String)):
                    parts.append(str(item))
            return "".join(parts)
        if op_name == "'":
            return str(operands[0])
        if op_name == '"':
            return str(operands[2])
    except Exception:
        return ""
    return ""


def normalize_clip_text(text: str) -> str:
    """
    Normalize text for matching clipping events to spans.
    """
    if not text:
        return ""
    cleaned = text.replace("\xa0", " ")
    return " ".join(cleaned.strip().split())


def _walk_content(
    pdf: pikepdf.Pdf,
    resources: Any,
    content_stream,
    results: list,
    page_num: int,
    xobject_stack: list,
    inherited_clip: bool = False,
):
    """
    Walk a content stream, tracking clipping via q/Q and W/W*.
    Recurse into Form XObjects encountered via 'Do' operator.
    """
    state = GraphicsState(clipped=inherited_clip)
    stack = []

    for operands, operator in _iter_content_operations(content_stream):
        try:
            op_name = operator.name  # type: ignore[attr-defined]
        except Exception:
            op_name = str(operator)
        if isinstance(op_name, str) and op_name.startswith("/"):
            op_name = op_name[1:]
        op_name = str(op_name)

        if op_name == "q":
            stack.append(state.clone())
        elif op_name == "Q":
            if stack:
                state = stack.pop()
            else:
                # Defensive: stray Q should not crash us
                state = GraphicsState(clipped=False)

        elif op_name in CLIP_OPS:
            # A clipping path is established for subsequent painting in THIS state.
            state.clipped = True

        elif op_name in TEXT_SHOW_OPS:
            # We are painting text now; record whether clip is active.
            text_snippet = extract_text_from_operands(op_name, operands)
            results.append({
                "page": page_num,
                "has_clip": state.clipped,
                "text": text_snippet,
                "operator": op_name,
                "xobject_context": list(xobject_stack),
            })

        elif op_name == "Do":
            # Invoke XObject (could be an image or a Form XObject that itself contains text).
            xname = operands[0]
            if not isinstance(xname, pikepdf.Name):
                continue
            xobj = _get_xobject(resources, xname) if resources is not None else None
            if not isinstance(xobj, pikepdf.Stream):
                continue

            subtype = xobj.get("/Subtype", None)
            if subtype == pikepdf.Name("/Form"):
                child_resources = _xobject_resources(xobj, resources)
                # Recurse into the form's content stream
                xobject_stack.append(str(xname))
                try:
                    _walk_content(
                        pdf=pdf,
                        resources=child_resources,
                        content_stream=xobj,
                        results=results,
                        page_num=page_num,
                        xobject_stack=xobject_stack,
                        inherited_clip=state.clipped,  # clipping survives into nested content
                    )
                finally:
                    xobject_stack.pop()


def analyze_pdf_hasclip(pdf_path: Path) -> Dict[int, list]:
    """
    Analyze an entire PDF and return detected text painting events per page with clipping info.

    Returns:
        Dictionary: {page_num: [{"text": str, "has_clip": bool, ...}, ...]}
    """
    if not HAS_PIKEPDF:
        return {}
    if PIKEPDF_PARSE_CONTENT is None:
        return {}

    try:
        pdf = pikepdf.open(pdf_path)
        results = []

        for i, page in enumerate(pdf.pages, start=1):
            resources = page.get("/Resources", None)
            _walk_content(
                pdf=pdf,
                resources=resources,
                content_stream=page.obj.get("/Contents"),
                results=results,
                page_num=i,
                xobject_stack=[],
                inherited_clip=False,
            )

        pdf.close()

        # Aggregate events per page
        page_events: Dict[int, list] = {}
        for r in results:
            page_num = r["page"]
            page_events.setdefault(page_num, []).append(r)

        return page_events

    except Exception:
        return {}


def calculate_effective_dpi(pdf_path: Path) -> Optional[float]:
    """
    Calculate the effective DPI of images in a PDF.

    Returns the median DPI across all images that cover a significant portion of pages.
    This helps determine the optimal rendering DPI for feature extraction.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Median effective DPI, or None if no suitable images found
    """
    try:
        doc = fitz.open(pdf_path)
        dpis = []

        for pno, page in enumerate(doc, start=1):
            # get_images(full=True) lists images; we'll get where each is drawn
            for img in page.get_images(full=True):
                xref = img[0]
                px_w, px_h = img[2], img[3]

                # One image can be drawn multiple times; get all placements
                try:
                    rects = page.get_image_rects(xref)  # list of fitz.Rect
                except AttributeError:
                    # Fallback for older PyMuPDF versions
                    try:
                        rects = page.get_image_bbox(xref)
                    except:
                        continue

                for rect in rects:
                    # Calculate physical dimensions in inches (72 points = 1 inch)
                    w_in = rect.width / 72.0
                    h_in = rect.height / 72.0

                    # Calculate DPI
                    if w_in > 0 and h_in > 0:
                        dpi_x = px_w / w_in
                        dpi_y = px_h / h_in

                        # Check if this image covers a significant portion of the page
                        # (either contains the page or page contains it)
                        covers_page = rect.contains(page.rect) or page.rect.contains(rect)

                        # Only include images that cover significant portions of pages
                        if covers_page or (rect.width > page.rect.width * 0.5 and rect.height > page.rect.height * 0.5):
                            # Use the average of horizontal and vertical DPI
                            avg_dpi = (dpi_x + dpi_y) / 2.0
                            dpis.append(avg_dpi)

        doc.close()

        if dpis:
            # Return median DPI (more robust than mean for outliers)
            dpis.sort()
            median_dpi = dpis[len(dpis) // 2]
            return median_dpi
        else:
            return None

    except Exception as e:
        # If calculation fails, return None to use default
        return None


def get_pdf_files() -> list:
    """Get list of PDF files from the test samples directory."""
    try:
        pdf_files = sorted([f.name for f in TEST_SAMPLES_PATH.glob("*.pdf")])
        return pdf_files
    except Exception as e:
        st.error(f"Error accessing test samples directory: {e}")
        return []


def check_pdf_has_text(pdf_path: Path) -> Tuple[bool, int, int, Dict[int, str]]:
    """
    Check if PDF has searchable text (OCRed).

    Returns:
        Tuple of (has_text, total_pages, pages_with_text, text_preview_by_page)
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_with_text = 0
        text_preview = {}

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text().strip()

            if text:
                pages_with_text += 1
                # Store first 200 characters of text as preview
                text_preview[page_num + 1] = text[:200] + ("..." if len(text) > 200 else "")

        doc.close()

        has_text = pages_with_text > 0
        return has_text, total_pages, pages_with_text, text_preview

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return False, 0, 0, {}


def check_pdf_restrictions(pdf_path: Path) -> Dict[str, any]:
    """
    Check PDF for security restrictions.

    Returns:
        Dictionary containing restriction information
    """
    try:
        # First try with pikepdf for detailed permissions
        if HAS_PIKEPDF:
            try:
                pdf = pikepdf.open(pdf_path)

                restrictions = {
                    "is_encrypted": pdf.is_encrypted,
                    "needs_password": False,  # If we can open it, we don't need password
                    "can_print": pdf.allow.print_highres or pdf.allow.print_lowres,
                    "can_copy": pdf.allow.extract,
                    "can_modify": pdf.allow.modify_form or pdf.allow.modify_other,
                    "can_annotate": pdf.allow.modify_annotation,
                }

                # Determine if there are any restrictions
                has_restrictions = (
                    not restrictions["can_print"] or
                    not restrictions["can_copy"] or
                    not restrictions["can_modify"] or
                    not restrictions["can_annotate"]
                )

                restrictions["has_restrictions"] = has_restrictions

                if has_restrictions:
                    restriction_list = []
                    if not restrictions["can_print"]:
                        restriction_list.append("Printing")
                    if not restrictions["can_copy"]:
                        restriction_list.append("Copying/Extracting text")
                    if not restrictions["can_modify"]:
                        restriction_list.append("Modifying content")
                    if not restrictions["can_annotate"]:
                        restriction_list.append("Adding annotations")

                    restrictions["restriction_note"] = f"Restricted: {', '.join(restriction_list)}"
                else:
                    restrictions["restriction_note"] = "No restrictions detected."

                pdf.close()
                return restrictions

            except pikepdf.PasswordError:
                return {
                    "is_encrypted": True,
                    "needs_password": True,
                    "has_restrictions": True,
                    "restriction_note": "Password required to open document."
                }
            except Exception as e:
                # Fall back to PyMuPDF
                pass

        # Fallback to PyMuPDF (less detailed)
        doc = fitz.open(pdf_path)

        restrictions = {
            "is_encrypted": doc.is_encrypted,
            "needs_password": doc.needs_pass,
            "can_print": True,
            "can_copy": True,
            "can_modify": True,
            "can_annotate": True,
        }

        # If document has encryption/security, check permissions
        if doc.is_encrypted:
            restrictions["has_restrictions"] = True
            restrictions["restriction_note"] = "Document is encrypted. Specific permissions may be restricted."
        else:
            restrictions["has_restrictions"] = False
            restrictions["restriction_note"] = "No encryption or restrictions detected."

        doc.close()
        return restrictions

    except Exception as e:
        st.error(f"Error checking PDF restrictions: {e}")
        return {
            "is_encrypted": False,
            "needs_password": False,
            "has_restrictions": False,
            "restriction_note": f"Error checking restrictions: {e}"
        }


def get_payload_path(pdf_path: Path) -> Path:
    """Get the path where the Docling payload should be saved."""
    from datetime import datetime
    doc_name = pdf_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save to Docling payloads folder with timestamp
    return PAYLOADS_PATH / f"{timestamp}_{doc_name}_Docling_raw_payload.json"


def check_payload_exists(pdf_path: Path) -> Tuple[bool, Optional[Path], Optional[Dict]]:
    """
    Check if a Docling payload already exists for the PDF.

    Returns:
        Tuple of (exists, payload_path, payload_info)
        payload_info contains size and modification time if exists
    """
    # Search for existing payloads with this document name
    doc_name = pdf_path.stem
    existing_payloads = sorted(PAYLOADS_PATH.glob(f"*{doc_name}_Docling_raw_payload.json"))

    if existing_payloads:
        # Return the most recent payload
        payload_path = existing_payloads[-1]
        stats = payload_path.stat()
        payload_info = {
            "size_kb": stats.st_size / 1024,
            "modified": time.ctime(stats.st_mtime)
        }
        return True, payload_path, payload_info
    else:
        # Return a new timestamped path for future generation
        payload_path = get_payload_path(pdf_path)
        return False, payload_path, None


def get_latest_docling_payload_path(doc_stem: str) -> Optional[Path]:
    """Return the most recent Docling payload path for a given document stem."""
    payloads = sorted(PAYLOADS_PATH.glob(f"*{doc_stem}_Docling_raw_payload.json"))
    return payloads[-1] if payloads else None


def get_latest_visible_payload_path(pdf_path: Path) -> Optional[Path]:
    """
    Return the most recent Docling payload for the flattened (OCRed) version of the PDF.
    """
    ocred_stem = f"{pdf_path.stem}_ocred"
    return get_latest_docling_payload_path(ocred_stem)


def serialize_cells_from_page(page) -> list:
    """
    Extract and serialize cell data from a Docling page.
    Captures word-level information including font metadata and color.

    Args:
        page: A docling.datamodel.base_models.Page object

    Returns:
        List of cell dictionaries with text, bbox, font info, color, etc.
    """
    cells_data = []

    if not hasattr(page, 'cells'):
        return cells_data

    for cell in page.cells:
        try:
            # Convert cell to dict using Pydantic's model_dump
            if hasattr(cell, 'model_dump'):
                cell_dict = cell.model_dump()
            elif hasattr(cell, 'dict'):
                cell_dict = cell.dict()
            else:
                # Fallback: manually extract key attributes
                cell_dict = {
                    'text': getattr(cell, 'text', ''),
                    'rect': getattr(cell, 'rect', None),
                    'index': getattr(cell, 'index', None),
                    'font_name': getattr(cell, 'font_name', None),
                    'font_key': getattr(cell, 'font_key', None),
                    'rgba': getattr(cell, 'rgba', None),
                    'confidence': getattr(cell, 'confidence', None),
                    'from_ocr': getattr(cell, 'from_ocr', None),
                    'text_direction': getattr(cell, 'text_direction', None),
                    'rendering_mode': getattr(cell, 'rendering_mode', None),
                    'widget': getattr(cell, 'widget', None),
                }

            cells_data.append(cell_dict)

        except Exception as e:
            # Skip cells that fail to serialize
            continue

    return cells_data


def generate_docling_payload(pdf_path: Path, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Path]]:
    """
    Generate enhanced Docling payload with word-level and font information.

    This creates a comprehensive payload with:
    - Document structure (standard Docling export)
    - Cell-level metadata (word bboxes, font info, colors)

    Args:
        pdf_path: Path to the PDF file
        force_regenerate: If True, regenerate even if payload exists

    Returns:
        Tuple of (success, message, payload_path)
    """
    if not HAS_DOCLING:
        return False, "Docling is not installed. Please install: pip install docling", None

    # Create payloads directory if it doesn't exist
    PAYLOADS_PATH.mkdir(parents=True, exist_ok=True)

    # Check if recent payload exists (within last hour)
    doc_name = pdf_path.stem
    existing_payloads = sorted(PAYLOADS_PATH.glob(f"*{doc_name}_Docling_raw_payload.json"))
    if existing_payloads and not force_regenerate:
        most_recent = existing_payloads[-1]
        return True, f"Docling payload already exists: {most_recent.name}", most_recent

    # Generate new timestamped payload path
    payload_path = get_payload_path(pdf_path)

    try:

        # Configure Docling pipeline with OCR, table structure, and parsed pages
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_parsed_pages = True  # KEY: Enables cell-level word data

        # Configure format options
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        # Create converter
        converter = DocumentConverter(format_options=format_options)

        # Convert document
        start_time = time.time()
        result = converter.convert(str(pdf_path))
        elapsed_time = time.time() - start_time

        # Get document from result
        dl_doc = getattr(result, "document", None) or result

        # Export document structure (standard)
        if hasattr(dl_doc, "export_to_dict"):
            document_payload = dl_doc.export_to_dict()
        elif hasattr(dl_doc, "export_to_json"):
            document_payload_json = dl_doc.export_to_json()
            if isinstance(document_payload_json, str):
                document_payload = json.loads(document_payload_json)
            else:
                document_payload = document_payload_json
        else:
            return False, "Document object has no export method", None

        # Extract cell-level metadata from result.pages (word-level with font info)
        cells_by_page = {}
        total_cells = 0

        if hasattr(result, 'pages') and result.pages:
            for page in result.pages:
                page_no = page.page_no
                cells_data = serialize_cells_from_page(page)

                cells_by_page[str(page_no)] = {
                    'page_no': page_no,
                    'size': page.size.model_dump() if hasattr(page.size, 'model_dump') else {
                        'width': page.size.width,
                        'height': page.size.height
                    },
                    'cells': cells_data,
                    'num_cells': len(cells_data)
                }
                total_cells += len(cells_data)

        # Create enhanced payload with metadata for feature detection
        enhanced_payload = {
            'metadata': {
                'num_pages': len(result.pages) if hasattr(result, 'pages') and result.pages else 0,
                'total_cells': total_cells,
                'has_parsed_pages': len(cells_by_page) > 0,  # Enables Hidden Text Detection
                'generation_time_seconds': elapsed_time
            },
            'docling_document': document_payload,
            'parsed_pages': cells_by_page
        }

        # Save enhanced payload with metadata
        with open(payload_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_payload, f, indent=2, ensure_ascii=False)

        file_size_kb = payload_path.stat().st_size / 1024
        message = f"Docling RAW payload generated in {elapsed_time:.1f}s ({file_size_kb:.1f} KB, {total_cells} cells)"
        return True, message, payload_path

    except Exception as e:
        return False, f"Error generating payload: {str(e)}", None


def load_payload_metadata(payload_path: Path) -> Optional[Dict]:
    """
    Load metadata from enhanced Docling payload.

    Returns:
        Dictionary with payload metadata or None if error
    """
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        metadata = {
            "file_size_kb": payload_path.stat().st_size / 1024,
            "top_level_keys": list(payload.keys()) if isinstance(payload, dict) else [],
        }

        # Check if this is an enhanced payload with parsed_pages
        if isinstance(payload, dict):
            # Enhanced payload structure
            if "metadata" in payload and isinstance(payload["metadata"], dict):
                meta = payload["metadata"]
                metadata["num_pages"] = meta.get("num_pages", "N/A")
                metadata["num_cells"] = meta.get("total_cells", "N/A")
                metadata["has_word_level"] = meta.get("has_parsed_pages", False)
                metadata["generation_time"] = meta.get("generation_time_seconds", "N/A")

            # Get info from docling_document if available
            if "docling_document" in payload:
                doc = payload["docling_document"]
                if "texts" in doc:
                    metadata["num_text_elements"] = len(doc["texts"]) if isinstance(doc["texts"], list) else "N/A"
                if "tables" in doc:
                    metadata["num_tables"] = len(doc["tables"]) if isinstance(doc["tables"], list) else "N/A"

            # Fallback: old format (just document structure)
            elif "pages" in payload:
                metadata["num_pages"] = len(payload["pages"]) if isinstance(payload["pages"], list) else "N/A"
                if "texts" in payload:
                    metadata["num_text_elements"] = len(payload["texts"]) if isinstance(payload["texts"], list) else "N/A"
                if "tables" in payload:
                    metadata["num_tables"] = len(payload["tables"]) if isinstance(payload["tables"], list) else "N/A"
                metadata["has_word_level"] = False

        return metadata

    except Exception as e:
        return None


def analyze_fonts_in_payload(payload_path: Path) -> Optional[Dict]:
    """
    Analyze font usage in enhanced Docling payload.

    Returns:
        Dictionary with:
        - font_stats: Dict[font_name, count]
        - font_items: Dict[font_name, List[Dict]] - all text items per font
        Or None if error or no parsed_pages available
    """
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        # Check if this is an enhanced payload with parsed_pages
        if not isinstance(payload, dict) or "parsed_pages" not in payload:
            return None

        parsed_pages = payload["parsed_pages"]

        # Dictionary to store font statistics and items
        font_stats = {}  # font_name -> count
        font_items = {}  # font_name -> list of text items with metadata

        # Iterate through all pages and cells
        for page_key, page_data in parsed_pages.items():
            if not isinstance(page_data, dict) or "cells" not in page_data:
                continue

            page_no = page_data.get("page_no", page_key)
            cells = page_data["cells"]

            for cell_idx, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    continue

                font_name = cell.get("font_name", "Unknown")
                text = cell.get("text", "").strip()

                # Skip empty cells
                if not text:
                    continue

                # Update statistics
                if font_name not in font_stats:
                    font_stats[font_name] = 0
                    font_items[font_name] = []

                font_stats[font_name] += 1

                # Store cell info with text
                font_items[font_name].append({
                    "page": page_no,
                    "text": text,
                    "font_key": cell.get("font_key"),
                    "rgba": cell.get("rgba"),
                    "confidence": cell.get("confidence"),
                    "from_ocr": cell.get("from_ocr"),
                    "rect": cell.get("rect")
                })

        # Sort font_stats by count (descending)
        font_stats = dict(sorted(font_stats.items(), key=lambda x: x[1], reverse=True))

        return {
            "font_stats": font_stats,
            "font_items": font_items,
            "total_fonts": len(font_stats),
            "total_text_items": sum(font_stats.values())
        }

    except Exception as e:
        st.error(f"Error analyzing fonts: {e}")
        return None


ROUNDING_CLASSIFICATION_LABELS = {
    "regular": "Regular size",
    "irregular": "Irregular size",
}


def format_rounding_classification(label: str) -> str:
    """
    Return a human-readable label for regular/irregular classifications.
    """
    return ROUNDING_CLASSIFICATION_LABELS.get(label, label)


def is_regular_measurement(value: float, tolerance: float = 0.0001) -> bool:
    """
    Check if a measurement should be considered "regular".

    Regular values include those that round to X.00, X.25, X.50, or X.75 when
    limited to four decimal places.

    Returns:
        True if the measurement matches a regular increment.
    """
    if not isinstance(value, (int, float)):
        return False

    rounded = round(float(value), 4)
    fractional = abs(rounded - math.floor(rounded))

    return (
        math.isclose(fractional, 0.0, abs_tol=tolerance)
        or math.isclose(fractional, 0.25, abs_tol=tolerance)
        or math.isclose(fractional, 0.5, abs_tol=tolerance)
        or math.isclose(fractional, 0.75, abs_tol=tolerance)
    )


def _iter_adobe_elements(node: Any, inherited_page: Optional[int] = None):
    """
    Yield (element_dict, page_number) pairs for an Adobe payload element and its children.
    """
    if not isinstance(node, dict):
        return

    current_page = node.get("Page", inherited_page)
    yield node, current_page

    kids = node.get("Kids")
    if isinstance(kids, list):
        for child in kids:
            yield from _iter_adobe_elements(child, current_page)


def analyze_adobe_payload(payload_path: Path) -> Optional[Dict]:
    """
    Analyze Adobe payload for font properties.

    Returns:
        Dictionary with stats and items for: embedded, encoding, family_name, name,
        monospaced, subset, HasClip, TextSize, LineHeight
    """
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or "elements" not in payload:
            return None

        elements = payload["elements"]
        if not isinstance(elements, list):
            return None

        # Dictionaries to track statistics
        embedded_stats = {}
        encoding_stats = {}
        family_name_stats = {}
        name_stats = {}
        monospaced_stats = {}
        subset_stats = {}
        has_clip_stats = {}
        text_size_stats = {}
        line_height_stats = {}

        # Dictionaries to store text samples
        embedded_items = {}
        encoding_items = {}
        family_name_items = {}
        name_items = {}
        monospaced_items = {}
        subset_items = {}
        has_clip_items = {}
        text_size_items = {}
        line_height_items = {}

        # Process each element
        for elem in elements:
            for node, page in _iter_adobe_elements(elem):
                if not isinstance(node, dict):
                    continue

                # Get text content (preserve empty strings but normalize type)
                text_raw = node.get("Text")
                if isinstance(text_raw, str):
                    text = text_raw.strip()
                    text_for_items = text_raw
                elif text_raw is None:
                    text = ""
                    text_for_items = ""
                else:
                    text_for_items = str(text_raw)
                    text = text_for_items.strip()

                # Get Font properties
                font = node.get("Font", {})
                if not isinstance(font, dict):
                    font = {}

                # Extract properties
                embedded = font.get("embedded")
                encoding = font.get("encoding")
                family_name = font.get("family_name")
                name = font.get("name")
                monospaced = font.get("monospaced")
                subset = font.get("subset")
                has_clip = node.get("HasClip")
                text_size_raw = node.get("TextSize")
                line_height_raw = node.get("LineHeight")

                if line_height_raw is None:
                    attributes = node.get("attributes")
                    if isinstance(attributes, dict):
                        line_height_raw = attributes.get("LineHeight")

                # Get page number (0-indexed)
                page_number = page if page is not None else 0

                # Get BBox for annotation rendering
                bbox = node.get("BBox")

                # Process embedded
                if embedded is not None:
                    embedded_key = str(embedded)
                    embedded_stats[embedded_key] = embedded_stats.get(embedded_key, 0) + 1
                    embedded_items.setdefault(embedded_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process encoding
                if encoding is not None:
                    encoding_key = str(encoding) if encoding else "None"
                    encoding_stats[encoding_key] = encoding_stats.get(encoding_key, 0) + 1
                    encoding_items.setdefault(encoding_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process family_name
                if family_name is not None:
                    family_key = str(family_name) if family_name else "Unknown"
                    family_name_stats[family_key] = family_name_stats.get(family_key, 0) + 1
                    family_name_items.setdefault(family_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process HasClip
                if has_clip is not None:
                    has_clip_key = str(has_clip)
                    has_clip_stats[has_clip_key] = has_clip_stats.get(has_clip_key, 0) + 1
                    has_clip_items.setdefault(has_clip_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process name
                if name is not None:
                    name_key = str(name) if name else "Unknown"
                    name_stats[name_key] = name_stats.get(name_key, 0) + 1
                    name_items.setdefault(name_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process monospaced
                if monospaced is not None:
                    monospaced_key = str(monospaced)
                    monospaced_stats[monospaced_key] = monospaced_stats.get(monospaced_key, 0) + 1
                    monospaced_items.setdefault(monospaced_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process subset
                if subset is not None:
                    subset_key = str(subset)
                    subset_stats[subset_key] = subset_stats.get(subset_key, 0) + 1
                    subset_items.setdefault(subset_key, []).append({
                        "text": text_for_items,
                        "page": page_number,
                        "BBox": bbox
                    })

                # Process TextSize with regular vs irregular classification
                if text_size_raw is not None:
                    try:
                        text_size_value = float(text_size_raw)

                        classification_key = "regular" if is_regular_measurement(text_size_value) else "irregular"

                        text_size_stats[classification_key] = text_size_stats.get(classification_key, 0) + 1
                        text_size_items.setdefault(classification_key, []).append({
                            "text": text_for_items,
                            "exact_size": text_size_value,
                            "classification": classification_key,
                            "classification_label": format_rounding_classification(classification_key),
                            "page": page_number,
                            "BBox": bbox
                        })
                    except (ValueError, TypeError):
                        pass

                # Process LineHeight with regular vs irregular classification
                if line_height_raw is not None:
                    try:
                        line_height_value = float(line_height_raw)

                        classification_key = "regular" if is_regular_measurement(line_height_value) else "irregular"

                        line_height_stats[classification_key] = line_height_stats.get(classification_key, 0) + 1
                        line_height_items.setdefault(classification_key, []).append({
                            "text": text_for_items,
                            "exact_height": line_height_value,
                            "classification": classification_key,
                            "classification_label": format_rounding_classification(classification_key),
                            "page": page_number,
                            "BBox": bbox
                        })
                    except (ValueError, TypeError):
                        pass

        # Sort statistics by count (descending)
        embedded_stats = dict(sorted(embedded_stats.items(), key=lambda x: x[1], reverse=True))
        encoding_stats = dict(sorted(encoding_stats.items(), key=lambda x: x[1], reverse=True))
        family_name_stats = dict(sorted(family_name_stats.items(), key=lambda x: x[1], reverse=True))
        name_stats = dict(sorted(name_stats.items(), key=lambda x: x[1], reverse=True))
        monospaced_stats = dict(sorted(monospaced_stats.items(), key=lambda x: x[1], reverse=True))
        subset_stats = dict(sorted(subset_stats.items(), key=lambda x: x[1], reverse=True))
        has_clip_stats = dict(sorted(has_clip_stats.items(), key=lambda x: x[1], reverse=True))
        text_size_stats = dict(sorted(text_size_stats.items(), key=lambda x: x[1], reverse=True))
        line_height_stats = dict(sorted(line_height_stats.items(), key=lambda x: x[1], reverse=True))

        return {
            "embedded": {
                "stats": embedded_stats,
                "items": embedded_items,
                "total": sum(embedded_stats.values())
            },
            "encoding": {
                "stats": encoding_stats,
                "items": encoding_items,
                "total": sum(encoding_stats.values())
            },
            "family_name": {
                "stats": family_name_stats,
                "items": family_name_items,
                "total": sum(family_name_stats.values())
            },
            "name": {
                "stats": name_stats,
                "items": name_items,
                "total": sum(name_stats.values())
            },
            "monospaced": {
                "stats": monospaced_stats,
                "items": monospaced_items,
                "total": sum(monospaced_stats.values())
            },
            "subset": {
                "stats": subset_stats,
                "items": subset_items,
                "total": sum(subset_stats.values())
            },
            "has_clip": {
                "stats": has_clip_stats,
                "items": has_clip_items,
                "total": sum(has_clip_stats.values())
            },
            "text_size": {
                "stats": text_size_stats,
                "items": text_size_items,
                "total": sum(text_size_stats.values())
            },
            "line_height": {
                "stats": line_height_stats,
                "items": line_height_items,
                "total": sum(line_height_stats.values())
            }
        }

    except Exception as e:
        st.error(f"Error analyzing Adobe payload: {e}")
        return None


def extract_bbox_coords(bbox: Dict) -> Tuple[float, float, float, float]:
    """
    Extract axis-aligned bounding box coordinates from various bbox formats.

    Supports:
    - Rotated rectangle format: {r_x0, r_y0, r_x1, r_y1, r_x2, r_y2, r_x3, r_y3}
    - Simple format: {x0, y0, x1, y1}

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    # Check if it's the rotated rectangle format (4 corners)
    if 'r_x0' in bbox:
        # Extract all 4 corners
        x_coords = [
            bbox.get('r_x0', 0),
            bbox.get('r_x1', 0),
            bbox.get('r_x2', 0),
            bbox.get('r_x3', 0)
        ]
        y_coords = [
            bbox.get('r_y0', 0),
            bbox.get('r_y1', 0),
            bbox.get('r_y2', 0),
            bbox.get('r_y3', 0)
        ]

        # Get axis-aligned bounding box (min/max of all corners)
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        return x_min, y_min, x_max, y_max

    # Check if it's the simple format (x0, y0, x1, y1)
    elif 'x0' in bbox:
        return bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']

    # Check if it's Docling table format (l, t, r, b)
    elif all(key in bbox for key in ('l', 't', 'r', 'b')):
        x0 = float(bbox['l'])
        y0 = float(min(bbox['t'], bbox['b']))
        x1 = float(bbox['r'])
        y1 = float(max(bbox['t'], bbox['b']))
        return x0, y0, x1, y1

    else:
        raise KeyError(f"Unknown bbox format, keys: {list(bbox.keys())}")


def normalize_cell_bbox(cell: Dict[str, Any]) -> Tuple[Dict[str, float], bool, str]:
    """
    Normalize a cell's bounding box into x0/y0/x1/y1 format and report zero status.

    Returns:
        (normalized_bbox, is_zero_bbox, source_key)
    """
    bbox_source = "missing"
    raw_bbox: Any = None

    for key in ("rect", "bbox"):
        value = cell.get(key)
        if value is not None:
            raw_bbox = value
            bbox_source = key
            break

    normalized_bbox = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}

    if isinstance(raw_bbox, dict):
        try:
            x0, y0, x1, y1 = extract_bbox_coords(raw_bbox)
            normalized_bbox = {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
            }
        except (KeyError, TypeError, ValueError):
            pass
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        x0, y0, x1, y1 = raw_bbox[:4]
        normalized_bbox = {
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
        }

    has_non_zero = any(abs(value) > 1e-6 for value in normalized_bbox.values())
    is_zero_bbox = not has_non_zero

    return normalized_bbox, is_zero_bbox, bbox_source


def extract_table_cells_from_docling(docling_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract table cell bounding boxes from a Docling payload for visualization.

    Returns:
        List of dicts with page index (0-based) and normalized bbox.
    """
    table_annotations: List[Dict[str, Any]] = []

    if not isinstance(docling_payload, dict):
        return table_annotations

    doc_struct = docling_payload.get("docling_document", {})
    tables = doc_struct.get("tables", [])

    if not isinstance(tables, list):
        return table_annotations

    seen_cells = set()

    for table in tables:
        if not isinstance(table, dict):
            continue

        prov_entries = table.get("prov", [])
        page_index: Optional[int] = None

        if isinstance(prov_entries, list):
            for prov in prov_entries:
                if not isinstance(prov, dict):
                    continue
                page_no = prov.get("page_no")
                if page_no is None:
                    continue
                try:
                    page_no = int(page_no)
                except (TypeError, ValueError):
                    continue
                page_index = page_no - 1 if page_no > 0 else page_no
                break

        if page_index is None:
            continue

        data = table.get("data", {})
        cells_source = []

        grid = data.get("grid")
        if isinstance(grid, list) and grid:
            for row in grid:
                if isinstance(row, list):
                    cells_source.extend(row)
        elif isinstance(data.get("table_cells"), list):
            cells_source.extend(data["table_cells"])

        for cell in cells_source:
            if not isinstance(cell, dict):
                continue

            bbox = cell.get("bbox")
            if not isinstance(bbox, dict):
                continue

            try:
                x0, y0, x1, y1 = extract_bbox_coords(bbox)
            except Exception:
                continue

            key = (page_index, round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
            if key in seen_cells:
                continue
            seen_cells.add(key)

            table_annotations.append({
                "page": page_index,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            })

    return table_annotations


def build_alignment_metadata(docling_payload: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build lookup of Docling structural metadata (table cells, text labels) by page.
    Used to suppress alignment checks for headers/section markers.
    """
    metadata_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    if not isinstance(docling_payload, dict):
        return metadata_by_page

    doc_struct = docling_payload.get("docling_document")
    if not isinstance(doc_struct, dict):
        return metadata_by_page

    tables = doc_struct.get("tables", [])
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue

            prov_entries = table.get("prov", [])
            page_index: Optional[int] = None
            if isinstance(prov_entries, list):
                for prov in prov_entries:
                    if not isinstance(prov, dict):
                        continue
                    page_no = prov.get("page_no")
                    if page_no is None:
                        continue
                    try:
                        page_no = int(page_no)
                    except (TypeError, ValueError):
                        continue
                    page_index = page_no - 1 if page_no > 0 else page_no
                    break

            if page_index is None:
                continue

            data = table.get("data", {})
            cell_candidates: List[Dict[str, Any]] = []

            grid = data.get("grid")
            if isinstance(grid, list):
                for row in grid:
                    if isinstance(row, list):
                        cell_candidates.extend(c for c in row if isinstance(c, dict))
            elif isinstance(data.get("table_cells"), list):
                cell_candidates.extend(c for c in data["table_cells"] if isinstance(c, dict))

            for cell in cell_candidates:
                bbox = cell.get("bbox")
                if not isinstance(bbox, dict):
                    continue

                try:
                    x0, y0, x1, y1 = extract_bbox_coords(bbox)
                except Exception:
                    continue

                metadata_by_page[page_index].append({
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "column_header": bool(cell.get("column_header")),
                    "row_header": bool(cell.get("row_header")),
                    "row_section": bool(cell.get("row_section")),
                    "type": "table_cell"
                })

    texts = doc_struct.get("texts", [])
    if isinstance(texts, list):
        for text_item in texts:
            if not isinstance(text_item, dict):
                continue

            prov_entries = text_item.get("prov", [])
            if not isinstance(prov_entries, list):
                continue

            bbox = None
            page_index: Optional[int] = None
            for prov in prov_entries:
                if not isinstance(prov, dict):
                    continue
                page_no = prov.get("page_no")
                bbox_dict = prov.get("bbox")
                if bbox_dict is None:
                    continue
                try:
                    x0, y0, x1, y1 = extract_bbox_coords(bbox_dict)
                except Exception:
                    continue
                bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                if page_no is not None:
                    try:
                        page_no = int(page_no)
                    except (TypeError, ValueError):
                        page_no = None
                if page_no is not None:
                    page_index = page_no - 1 if page_no > 0 else page_no
                break

            if bbox is None or page_index is None:
                continue

            metadata_by_page[page_index].append({
                "bbox": bbox,
                "label": text_item.get("label"),
                "type": "text"
            })

    return metadata_by_page


def match_alignment_metadata(
    bbox: Dict[str, float],
    metadata_items: List[Dict[str, Any]],
    min_iou: float = 0.6
) -> Optional[Dict[str, Any]]:
    """
    Find the best Docling metadata entry overlapping the provided bbox.
    """
    best_item: Optional[Dict[str, Any]] = None
    best_iou: float = 0.0

    for item in metadata_items:
        item_bbox = item.get("bbox")
        if not item_bbox:
            continue
        try:
            overlap = calculate_iou(bbox, item_bbox)
        except Exception:
            continue
        if overlap > best_iou:
            best_iou = overlap
            best_item = item

    if best_iou >= min_iou:
        return best_item

    return None


def consolidate_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate cells by removing words that are contained within larger phrases/cells.

    Example: If "HOURS" is spatially contained in "HOURS /UNITS", keep only "HOURS /UNITS".

    Args:
        cells: List of cell dictionaries with 'text' and 'bbox' keys

    Returns:
        Filtered list of cells with contained words removed
    """
    if not cells:
        return []

    # Sort cells by area (descending) to process larger cells first
    sorted_cells = sorted(
        cells,
        key=lambda c: (c["bbox"]["x1"] - c["bbox"]["x0"]) * (c["bbox"]["y1"] - c["bbox"]["y0"]),
        reverse=True
    )

    consolidated = []
    removed_indices = set()

    for i, cell in enumerate(sorted_cells):
        if i in removed_indices:
            continue

        cell_bbox = cell["bbox"]
        cell_text = cell["text"].strip()

        # Check if this cell contains any smaller cells
        for j, other_cell in enumerate(sorted_cells[i+1:], start=i+1):
            if j in removed_indices:
                continue

            other_bbox = other_cell["bbox"]
            other_text = other_cell["text"].strip()

            # Check if other_cell is spatially contained within cell
            # Allow small tolerance for rounding errors (0.5pt)
            tolerance = 0.5
            x_contained = (other_bbox["x0"] >= cell_bbox["x0"] - tolerance and
                          other_bbox["x1"] <= cell_bbox["x1"] + tolerance)
            y_contained = (other_bbox["y0"] >= cell_bbox["y0"] - tolerance and
                          other_bbox["y1"] <= cell_bbox["y1"] + tolerance)

            # Check if the text of the smaller cell appears in the larger cell
            text_contained = other_text.lower() in cell_text.lower()

            if x_contained and y_contained and text_contained:
                # Mark the smaller cell for removal
                removed_indices.add(j)

        consolidated.append(cell)

    return consolidated


def detect_alignment_baselines(
    cells: List[Dict[str, Any]],
    page_width: float,
    page_metadata: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Detect alignment baselines (dominant left/right edges) in a page.

    Returns list of baselines with:
    - orientation: "left" or "right"
    - value: x-coordinate of the baseline
    - page_position: "page_left", "page_right", or "mid_page"
    - y_values: list of y-coordinates of blocks aligned to this baseline
    - count: number of items aligned to this baseline
    """
    if not cells:
        return []

    # Filter out structural elements using metadata
    filtered_cells = []
    for cell in cells:
        if page_metadata:
            match = match_alignment_metadata(cell["bbox"], page_metadata, min_iou=0.6)
            if match:
                # Skip table headers and section headers
                if match.get("column_header") or match.get("row_header"):
                    continue
                if match.get("label") == "section_header":
                    continue

        filtered_cells.append(cell)

    if not filtered_cells:
        return []

    # Cluster left and right edges at 0.1pt precision
    left_clusters: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    right_clusters: Dict[float, List[Dict[str, Any]]] = defaultdict(list)

    for cell in filtered_cells:
        bbox = cell["bbox"]
        left_edge = bbox["x0"]
        right_edge = bbox["x1"]

        # Round to 0.1pt for clustering
        left_key = round(left_edge, 1)
        right_key = round(right_edge, 1)

        left_clusters[left_key].append(cell)
        right_clusters[right_key].append(cell)

    baselines = []

    # Process left-aligned baselines
    for cluster_key, cluster_cells in left_clusters.items():
        if len(cluster_cells) < 3:  # Require at least 3 items
            continue

        # Calculate mean x-coordinate
        mean_x = sum(c["bbox"]["x0"] for c in cluster_cells) / len(cluster_cells)

        # Collect y-values (use center y for each block)
        y_values = [(c["bbox"]["y0"] + c["bbox"]["y1"]) / 2 for c in cluster_cells]

        # Determine page position
        if mean_x < page_width * 0.25:
            page_position = "page_left"
        elif mean_x > page_width * 0.75:
            page_position = "page_right"
        else:
            page_position = "mid_page"

        baselines.append({
            "orientation": "left",
            "value": mean_x,
            "page_position": page_position,
            "y_values": y_values,
            "count": len(cluster_cells)
        })

    # Process right-aligned baselines
    for cluster_key, cluster_cells in right_clusters.items():
        if len(cluster_cells) < 3:  # Require at least 3 items
            continue

        # Calculate mean x-coordinate
        mean_x = sum(c["bbox"]["x1"] for c in cluster_cells) / len(cluster_cells)

        # Collect y-values (use center y for each block)
        y_values = [(c["bbox"]["y0"] + c["bbox"]["y1"]) / 2 for c in cluster_cells]

        # Determine page position
        if mean_x < page_width * 0.25:
            page_position = "page_left"
        elif mean_x > page_width * 0.75:
            page_position = "page_right"
        else:
            page_position = "mid_page"

        baselines.append({
            "orientation": "right",
            "value": mean_x,
            "page_position": page_position,
            "y_values": y_values,
            "count": len(cluster_cells)
        })

    # Sort baselines by count (descending) then by value
    baselines.sort(key=lambda b: (-b["count"], b["value"]))

    return baselines


def detect_docling_alignment_anomalies(
    docling_payload: Dict[str, Any],
    target_labels: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Detect misalignment issues using baseline detection approach.

    NEW APPROACH (2025-11-05):
    1. Consolidate cells (remove words contained in larger phrases)
    2. Detect alignment baselines per page
    3. Future: Check for deviations from baselines

    OLD APPROACH (ARCHIVED):
    Previous direct label/value comparison logic has been archived.
    See docling_alignment_checks.md for details.

    Returns:
        List of anomalies with bounding boxes, page index, and textual reasons.
        Currently returns empty list while baseline detection is being implemented.
    """
    if not isinstance(docling_payload, dict):
        return []

    parsed_pages = docling_payload.get("parsed_pages")
    if not isinstance(parsed_pages, dict):
        return []

    alignment_metadata = build_alignment_metadata(docling_payload)

    # Store baselines for each page (will be used for rendering)
    baselines_by_page: Dict[int, List[Dict[str, Any]]] = {}

    anomalies: List[Dict[str, Any]] = []

    for page_key, page_data in parsed_pages.items():
        if not isinstance(page_data, dict):
            continue

        cells = page_data.get("cells", [])
        if not isinstance(cells, list) or not cells:
            continue

        page_no = page_data.get("page_no")
        if page_no is None:
            try:
                page_no = int(page_key)
            except (TypeError, ValueError):
                continue

        # Get page dimensions
        page_size = page_data.get("size", {})
        page_width = page_size.get("width", 612)  # Default to letter width if not found

        # Normalize cells
        normalized_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            text = (cell.get("text") or "").strip()
            if not text:
                continue

            rect = cell.get("rect") or cell.get("bbox")
            if not isinstance(rect, dict):
                continue

            try:
                x0, y0, x1, y1 = extract_bbox_coords(rect)
            except Exception:
                continue

            normalized_cells.append({
                "text": text,
                "text_lower": text.lower(),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "metadata": {}
            })

        if not normalized_cells:
            continue

        # Apply cell consolidation (remove contained words)
        consolidated_cells = consolidate_cells(normalized_cells)

        # Attach metadata to consolidated cells
        page_metadata = alignment_metadata.get(page_no, [])
        if page_metadata:
            for entry in consolidated_cells:
                match = match_alignment_metadata(entry["bbox"], page_metadata)
                if match:
                    entry["metadata"] = match

        # Detect alignment baselines for this page
        baselines = detect_alignment_baselines(consolidated_cells, page_width, page_metadata)
        baselines_by_page[page_no] = baselines

        # TODO: Future implementation - check for anomalies based on baselines
        # For now, we just detect and store baselines without flagging anomalies

    # Store baselines in the payload for rendering
    if baselines_by_page:
        docling_payload["_alignment_baselines"] = baselines_by_page

    # Return empty anomalies list for now (baseline-based anomaly detection to be implemented)
    # Baselines are stored in docling_payload["_alignment_baselines"] for visualization
    return anomalies


# =============================================================================


def detect_colon_spacing_anomalies(
    docling_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Detect spacing inconsistencies in label:value pairs separated by colons.

    This detects forgery by finding deviations from the dominant spacing pattern
    in colon-separated label-value pairs (e.g., "EMP.CODE: HM20927").

    Returns:
        List of anomalies with classification:
        - 'consistent': Green - matches dominant pattern
        - 'deviation': Red - spacing deviation (potential forgery)
        - 'right_aligned': Orange - right-aligned exception
    """
    import re

    if not isinstance(docling_payload, dict):
        return []

    parsed_pages = docling_payload.get("parsed_pages")
    if not isinstance(parsed_pages, dict):
        return []

    all_anomalies: List[Dict[str, Any]] = []

    for page_key, page_data in parsed_pages.items():
        if not isinstance(page_data, dict):
            continue

        cells = page_data.get("cells", [])
        if not isinstance(cells, list) or not cells:
            continue

        page_no = page_data.get("page_no")
        if page_no is None:
            try:
                page_no = int(page_key)
            except (TypeError, ValueError):
                continue

        # Normalize ALL cells (not just those with colons - we need values too!)
        normalized_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            text = (cell.get("text") or "").strip()
            if not text:
                continue

            rect = cell.get("rect") or cell.get("bbox")
            if not isinstance(rect, dict):
                continue

            try:
                x0, y0, x1, y1 = extract_bbox_coords(rect)
            except Exception:
                continue

            normalized_cells.append({
                "text": text,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            })

        if not normalized_cells:
            continue

        # Extract colon pairs
        colon_pairs = extract_colon_pairs(normalized_cells)

        if len(colon_pairs) < 2:  # Need at least 2 pairs to establish a pattern
            continue

        # Analyze spacing patterns
        spacing_analysis = analyze_colon_spacing(colon_pairs)

        if not spacing_analysis:
            continue

        dominant_spacing = spacing_analysis['dominant_spacing']

        # Detect right-aligned values
        right_aligned_indices = detect_right_aligned_values(colon_pairs)

        # Classify each pair
        for idx, pair in enumerate(colon_pairs):
            spacing = pair['spacing']
            deviation = abs(spacing - dominant_spacing)

            # Determine classification
            if idx in right_aligned_indices:
                # Right-aligned exception (orange)
                classification = 'right_aligned'
                reason = f"Right-aligned value (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt)"
            elif deviation <= COLON_SPACING_TOLERANCE:
                # Consistent with pattern (green)
                classification = 'consistent'
                reason = f"Consistent spacing (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt)"
            else:
                # Deviation from pattern (red - potential forgery)
                classification = 'deviation'
                reason = f"Spacing deviation (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt, deviation: {deviation:.1f}pt)"

            all_anomalies.append({
                'page': page_no,
                'label': pair['label'],
                'value': pair['value'],
                'bbox': pair['value_bbox'],
                'label_bbox': pair['label_bbox'],
                'classification': classification,
                'spacing': spacing,
                'dominant_spacing': dominant_spacing,
                'deviation': deviation,
                'reason': reason
            })

    return all_anomalies


def extract_colon_pairs(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract label:value pairs from cells containing colons.

    Handles two cases:
    1. Label and value in same cell: "LABEL: VALUE"
    2. Label and value in separate cells: "LABEL:" in one cell, "VALUE" in next cell

    Returns list of pairs with spacing information.
    """
    import re

    pairs = []

    for i, cell in enumerate(cells):
        text = cell['text']
        bbox = cell['bbox']

        # Skip if colon is part of time/date/URL
        if is_excluded_colon(text):
            continue

        # Case 1: Label ends with colon (separate cells)
        if text.endswith(':'):
            label_text = text
            label_bbox = bbox

            # Find value cell (next cell on same baseline, to the right)
            value_cell = None
            label_center_y = (bbox['y0'] + bbox['y1']) / 2

            for j in range(i + 1, len(cells)):
                candidate = cells[j]
                candidate_bbox = candidate['bbox']

                # Check if on same baseline (within 2pt)
                candidate_center_y = (candidate_bbox['y0'] + candidate_bbox['y1']) / 2
                if abs(candidate_center_y - label_center_y) > COLON_BASELINE_TOLERANCE:
                    continue

                # Check if to the right
                if candidate_bbox['x0'] <= bbox['x1']:
                    continue

                # Check if reasonably close
                horizontal_gap = candidate_bbox['x0'] - bbox['x1']
                if horizontal_gap > COLON_MAX_DISTANCE:
                    continue

                # Found potential value cell
                value_cell = candidate
                break

            if not value_cell:
                continue

            value_text = value_cell['text']
            value_bbox = value_cell['bbox']

            # Calculate spacing (from label right edge to value left edge)
            spacing = value_bbox['x0'] - label_bbox['x1']

            pairs.append({
                'label': label_text,
                'value': value_text,
                'label_bbox': label_bbox,
                'value_bbox': value_bbox,
                'spacing': spacing,
                'full_text': f"{label_text} {value_text}",
                'full_bbox': bbox
            })

        # Case 2: Label and value in same cell: "LABEL: VALUE"
        elif ':' in text:
            colon_idx = text.find(':')
            label_text = text[:colon_idx + 1].strip()
            value_text = text[colon_idx + 1:].strip()

            # Skip if no value after colon
            if not value_text:
                continue

            # Estimate positions (simplified - assumes monospace-ish layout)
            text_width = bbox['x1'] - bbox['x0']
            colon_ratio = (colon_idx + 1) / len(text)

            # Estimate colon position
            colon_x = bbox['x0'] + (text_width * colon_ratio)

            # Estimate value start position
            value_start_ratio = (colon_idx + 1 + (len(text) - len(text[colon_idx + 1:].lstrip()))) / len(text)
            value_x0 = bbox['x0'] + (text_width * value_start_ratio)

            # Calculate spacing (distance from colon to value start)
            spacing = value_x0 - colon_x

            # Create label and value bboxes (estimates)
            label_bbox = {
                'x0': bbox['x0'],
                'y0': bbox['y0'],
                'x1': colon_x,
                'y1': bbox['y1']
            }

            value_bbox = {
                'x0': value_x0,
                'y0': bbox['y0'],
                'x1': bbox['x1'],
                'y1': bbox['y1']
            }

            pairs.append({
                'label': label_text,
                'value': value_text,
                'label_bbox': label_bbox,
                'value_bbox': value_bbox,
                'spacing': spacing,
                'full_text': text,
                'full_bbox': bbox
            })

    return pairs


def is_excluded_colon(text: str) -> bool:
    """
    Check if text contains a colon that should be excluded (time, date, URL, ratio).
    """
    import re

    # Time patterns: HH:MM or HH:MM:SS
    if re.search(r'\d{1,2}:\d{2}(:\d{2})?', text):
        return True

    # URL patterns: http://, https://, mailto:
    if re.search(r'(https?|mailto|ftp):', text, re.IGNORECASE):
        return True

    # Ratio patterns: N:M (e.g., 16:9, 4:3)
    if re.search(r'\b\d+:\d+\b', text):
        return True

    return False


def is_excluded_value(text: str) -> bool:
    """
    Check if a value text should be excluded from colon pattern analysis.
    Excludes dates, times, URLs, etc.
    """
    import re

    # Date patterns: YYYY/MM/DD, DD/MM/YYYY, YYYY-MM-DD
    if re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', text):
        return True
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text):
        return True

    # Time patterns: HH:MM or HH:MM:SS
    if re.search(r'\d{1,2}:\d{2}(:\d{2})?', text):
        return True

    # URL patterns
    if re.search(r'(https?|mailto|ftp)://', text, re.IGNORECASE):
        return True

    return False


def analyze_colon_spacing(pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Analyze spacing patterns in colon pairs to find dominant spacing.

    Returns:
        Dict with dominant_spacing and cluster_info, or None if insufficient data
    """
    if len(pairs) < COLON_MIN_CLUSTER_SIZE:
        return None

    # Collect all spacing values
    spacings = [p['spacing'] for p in pairs]

    # Bucket spacings at 0.5pt precision
    spacing_buckets: Dict[float, List[float]] = defaultdict(list)
    for spacing in spacings:
        bucket_key = round(spacing * 2) / 2  # Round to nearest 0.5pt
        spacing_buckets[bucket_key].append(spacing)

    # Find dominant cluster (most frequent)
    if not spacing_buckets:
        return None

    dominant_bucket, dominant_items = max(
        spacing_buckets.items(),
        key=lambda kv: len(kv[1])
    )

    # Require minimum cluster size
    if len(dominant_items) < COLON_MIN_CLUSTER_SIZE:
        return None

    # Calculate mean spacing for dominant cluster
    dominant_spacing = sum(dominant_items) / len(dominant_items)

    return {
        'dominant_spacing': dominant_spacing,
        'dominant_count': len(dominant_items),
        'total_pairs': len(pairs)
    }


def detect_right_aligned_values(pairs: List[Dict[str, Any]]) -> set:
    """
    Detect which value bboxes are right-aligned (exception to spacing rules).

    Returns:
        Set of indices of right-aligned pairs
    """
    if len(pairs) < 2:  # Reduced from 3 to 2 for better detection
        return set()

    # Cluster right edges at 1.5pt precision (increased from 0.5pt for better tolerance)
    right_edge_buckets: Dict[float, List[int]] = defaultdict(list)

    for idx, pair in enumerate(pairs):
        right_edge = pair['value_bbox']['x1']
        bucket_key = round(right_edge * 0.67) / 0.67  # Round to nearest 1.5pt (1/0.67 ≈ 1.5)
        right_edge_buckets[bucket_key].append(idx)

    # Find clusters with ≥2 items (reduced from ≥3 for better detection)
    right_aligned_indices = set()
    for bucket_key, indices in right_edge_buckets.items():
        if len(indices) >= 2:  # Reduced from 3 to 2 for better detection
            right_aligned_indices.update(indices)

    return right_aligned_indices


def calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) for two bboxes.

    Bbox formats supported:
    - Rotated rectangle: {r_x0, r_y0, r_x1, r_y1, r_x2, r_y2, r_x3, r_y3, coord_origin}
    - Simple format: {x0, y0, x1, y1}
    """
    try:
        # Extract coordinates using format-aware helper
        x1_min, y1_min, x1_max, y1_max = extract_bbox_coords(bbox1)
        x2_min, y2_min, x2_max, y2_max = extract_bbox_coords(bbox2)

        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area
    except Exception as e:
        # Log bbox format issues for debugging
        print(f"[DEBUG] IoU calculation failed: {e}")
        print(f"[DEBUG] bbox1 keys: {list(bbox1.keys()) if isinstance(bbox1, dict) else 'not a dict'}")
        print(f"[DEBUG] bbox2 keys: {list(bbox2.keys()) if isinstance(bbox2, dict) else 'not a dict'}")
        return 0.0


def find_overlapping_pairs_optimized(
    cells: List[Dict],
    threshold: float = 0.25,
    cell_w: float = None,
    cell_h: float = None,
) -> List[Tuple[int, int, float]]:
    """
    Find overlapping cell pairs using spatial grid optimization.

    Reduces complexity from O(n²) to ~O(9cn) where c = avg cells per grid cell.

    Args:
        cells: List of cell dictionaries with 'rect' key containing bbox
        threshold: Minimum IoU threshold (default 0.25 = 25% overlap)
        cell_w: Grid cell width (auto-calculated from median if None)
        cell_h: Grid cell height (auto-calculated from median if None)

    Returns:
        List of (i, j, iou_score) tuples where i < j and iou_score >= threshold
    """
    from collections import defaultdict
    from statistics import median

    n = len(cells)
    if n == 0:
        return []

    # Extract all bboxes and convert to (x_min, y_min, x_max, y_max) format
    bboxes = []
    for cell in cells:
        rect = cell.get('rect', {})
        try:
            x_min, y_min, x_max, y_max = extract_bbox_coords(rect)
            bboxes.append((x_min, y_min, x_max, y_max))
        except:
            # Skip cells with invalid bboxes
            bboxes.append(None)

    # Filter out None bboxes
    valid_indices = [i for i, bbox in enumerate(bboxes) if bbox is not None]
    valid_bboxes = [bboxes[i] for i in valid_indices]

    if len(valid_bboxes) == 0:
        return []

    # Calculate widths/heights; use median for cell size (robust to outliers)
    widths = [max(1e-9, b[2] - b[0]) for b in valid_bboxes]
    heights = [max(1e-9, b[3] - b[1]) for b in valid_bboxes]
    cw = cell_w or max(1e-9, median(widths))
    ch = cell_h or max(1e-9, median(heights))

    # Global bounds to anchor grid indexing
    minx = min(b[0] for b in valid_bboxes)
    miny = min(b[1] for b in valid_bboxes)

    def cell_range_for_box(bbox):
        """Get inclusive cell indices touched by the box"""
        x0i = int((bbox[0] - minx) // cw)
        x1i = int((bbox[2] - minx) // cw)
        y0i = int((bbox[1] - miny) // ch)
        y1i = int((bbox[3] - miny) // ch)
        return x0i, x1i, y0i, y1i

    # Map cells -> indices of boxes that touch them
    grid = defaultdict(list)
    cell_ranges = []
    for local_idx, bbox in enumerate(valid_bboxes):
        xr = cell_range_for_box(bbox)
        cell_ranges.append(xr)
        x0i, x1i, y0i, y1i = xr
        for xi in range(x0i, x1i + 1):
            for yi in range(y0i, y1i + 1):
                grid[(xi, yi)].append(local_idx)

    results = []
    seen = set()  # avoid duplicate pairs

    # For each box, check only its cell + 8 neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    for local_i, bbox_i in enumerate(valid_bboxes):
        x0i, x1i, y0i, y1i = cell_ranges[local_i]

        # Collect candidate indices from this cell + neighbors
        cand_idxs = set()
        for xi in range(x0i, x1i + 1):
            for yi in range(y0i, y1i + 1):
                for dx, dy in neighbors:
                    cand_idxs.update(grid.get((xi + dx, yi + dy), []))

        for local_j in cand_idxs:
            if local_j <= local_i:  # upper triangle only
                continue

            key = (local_i, local_j)
            if key in seen:
                continue
            seen.add(key)

            bbox_j = valid_bboxes[local_j]

            # Quick 1D projection reject: if no 1D overlap in x or y, skip
            if bbox_i[2] <= bbox_j[0] or bbox_j[2] <= bbox_i[0]:  # no x-overlap
                continue
            if bbox_i[3] <= bbox_j[1] or bbox_j[3] <= bbox_i[1]:  # no y-overlap
                continue

            # Calculate IoU using existing function
            cell_i_rect = cells[valid_indices[local_i]].get('rect', {})
            cell_j_rect = cells[valid_indices[local_j]].get('rect', {})
            iou = calculate_iou(cell_i_rect, cell_j_rect)

            if iou >= threshold:
                # Map back to original indices
                orig_i = valid_indices[local_i]
                orig_j = valid_indices[local_j]
                results.append((orig_i, orig_j, iou))

    return results


def create_visible_only_pdf(input_pdf: Path, output_pdf: Path) -> Path:
    """
    Create a visible-only searchable PDF using PDF -> Image (400 DPI) -> OCR -> PDF pipeline.
    This removes all hidden text layers and adds a fresh OCR text layer using pytesseract.
    This ensures Docling can extract word-level information from the flattened version.

    Args:
        input_pdf: Path to the original PDF
        output_pdf: Path where the visible-only PDF should be saved

    Returns:
        Path to the created visible-only PDF
    """
    if not HAS_FLATTENING:
        raise ImportError("pdf2image and pytesseract required. Install: pip install pdf2image pytesseract")

    try:
        from PyPDF2 import PdfMerger
    except ImportError:
        raise ImportError("PyPDF2 required for PDF merging. Install: pip install PyPDF2")

    # Ensure PDF processing directory exists
    PDF_PROCESSING_PATH.mkdir(parents=True, exist_ok=True)

    # Generate consistent naming for intermediate files
    pdf_basename = input_pdf.stem

    # Convert PDF to images at 400 DPI
    images = convert_from_path(
        input_pdf,
        dpi=DPI,
        fmt='png'
    )

    # Process each image and create searchable PDF pages
    pdf_pages = []
    png_paths = []

    for i, img in enumerate(images):
        # Save image temporarily
        png_path = PDF_PROCESSING_PATH / f"{pdf_basename}_page{i:03d}_{DPI}dpi.png"
        img.save(png_path, 'PNG', dpi=(DPI, DPI))
        png_paths.append(png_path)

        # Create searchable PDF from image using pytesseract
        config_tokens: List[str] = [f'--dpi {DPI}']
        pdf_path = PDF_PROCESSING_PATH / f"{pdf_basename}_page{i:03d}_searchable.pdf"
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            img,
            extension='pdf',
            lang='eng',
            config=' '.join(config_tokens)
        )
        with pdf_path.open('wb') as f:
            f.write(pdf_bytes)
        pdf_pages.append(pdf_path)

    # Merge all PDF pages into one
    merger = PdfMerger()
    for pdf_page in pdf_pages:
        merger.append(str(pdf_page))
    merger.write(str(output_pdf))
    merger.close()

    # Cleanup intermediate files if not keeping them
    if not KEEP_INTERMEDIATE_FILES:
        for png_path in png_paths:
            if png_path.exists():
                png_path.unlink()
        for pdf_page in pdf_pages:
            if pdf_page.exists():
                pdf_page.unlink()

    return output_pdf


def find_cell_in_visible_payload(
    cell_text: str,
    cell_bbox: Dict,
    visible_cells: List[Dict],
    mode: str = "phrase"
) -> Tuple[bool, Optional[Dict], Optional[Dict[str, Any]]]:
    """
    Check if a cell from original payload exists in visible-only payload.

    Args:
        cell_text: Text from the original payload cell.
        cell_bbox: Bounding box dictionary from the original payload.
        visible_cells: List of cells from the visible-only payload (same page).
        mode: 'word' for word-level matching, 'phrase' otherwise.

    Returns:
        (found, matching_cell, debug_info)
    """
    normalized_target = normalize_text(cell_text)
    if not normalized_target:
        return False, None, None

    try:
        orig_x0, orig_y0, orig_x1, orig_y1 = extract_bbox_coords(cell_bbox)
    except Exception:
        return False, None, None

    orig_width = max(1e-6, orig_x1 - orig_x0)
    orig_height = max(1e-6, orig_y1 - orig_y0)
    orig_center = ((orig_x0 + orig_x1) / 2.0, (orig_y0 + orig_y1) / 2.0)
    max_vertical_gap = orig_height * 0.5  # fallback threshold per requirements
    candidate_iou_threshold = max(MIN_CANDIDATE_IOU, BBOX_TOLERANCE * 0.5)
    similarity_threshold = WORD_SIMILARITY_THRESHOLD if mode == "word" else PHRASE_SIMILARITY_THRESHOLD
    lenient_similarity_threshold = (
        LENIENT_WORD_SIMILARITY_THRESHOLD if mode == "word" else LENIENT_PHRASE_SIMILARITY_THRESHOLD
    )

    best_similarity = -1.0
    best_similarity_info: Optional[Dict[str, Any]] = None
    closest_distance = float("inf")
    closest_distance_info: Optional[Dict[str, Any]] = None

    for vcell in visible_cells:
        v_text = vcell.get("text", "")
        if not isinstance(v_text, str) or not v_text.strip():
            continue

        v_bbox = vcell.get("rect", {})
        try:
            vx0, vy0, vx1, vy1 = extract_bbox_coords(v_bbox)
        except Exception:
            continue

        normalized_visible = normalize_text(v_text)
        if not normalized_visible:
            continue

        iou = calculate_iou(cell_bbox, v_bbox)
        tokens = [normalize_text(token) for token in v_text.split() if normalize_text(token)]
        substring_hit = normalized_target in normalized_visible or normalized_visible in normalized_target

        # Compute similarity depending on matching mode
        if mode == "word":
            token_scores = [levenshtein_similarity(normalized_target, token) for token in tokens] or [0.0]
            token_scores.append(levenshtein_similarity(normalized_target, normalized_visible))
            similarity = max(token_scores)
        else:
            token_scores = [levenshtein_similarity(normalized_target, normalized_visible)]
            token_scores.extend(levenshtein_similarity(normalized_target, token) for token in tokens)
            similarity = max(token_scores)
            if substring_hit:
                similarity = 1.0

        lev_distance = levenshtein_distance(normalized_target, normalized_visible)
        vis_center = ((vx0 + vx1) / 2.0, (vy0 + vy1) / 2.0)
        center_distance = math.hypot(orig_center[0] - vis_center[0], orig_center[1] - vis_center[1])
        x_overlap = max(0.0, min(orig_x1, vx1) - max(orig_x0, vx0))
        if vy0 > orig_y1:
            vertical_gap = vy0 - orig_y1
        elif orig_y0 > vy1:
            vertical_gap = orig_y0 - vy1
        else:
            vertical_gap = 0.0

        candidate_info = {
            "text": v_text.strip(),
            "similarity": round(similarity, 4),
            "levenshtein_distance": lev_distance,
            "center_distance": round(center_distance, 4),
            "x_overlap": round(x_overlap, 4),
            "vertical_gap": round(vertical_gap, 4),
            "iou": round(iou, 4),
        }

        if similarity > best_similarity:
            best_similarity = similarity
            best_similarity_info = candidate_info

        if center_distance < closest_distance:
            closest_distance = center_distance
            closest_distance_info = candidate_info

        if iou >= candidate_iou_threshold and similarity >= similarity_threshold:
            candidate_info["match_reason"] = "primary_iou"
            return True, vcell, candidate_info

        if iou >= candidate_iou_threshold and similarity >= lenient_similarity_threshold:
            candidate_info["match_reason"] = "lenient_iou"
            return True, vcell, candidate_info

        if x_overlap > 0 and vertical_gap <= max_vertical_gap:
            if similarity >= similarity_threshold:
                candidate_info["match_reason"] = "primary_overlap"
                return True, vcell, candidate_info
            if similarity >= lenient_similarity_threshold:
                candidate_info["match_reason"] = "lenient_overlap"
                return True, vcell, candidate_info

    debug_info = best_similarity_info or closest_distance_info
    if debug_info:
        debug_info.setdefault("match_reason", "closest_candidate")

    return False, None, debug_info


def summarize_payload_cells(payload: Dict) -> Dict[str, float]:
    """
    Generate aggregate statistics about text cells in a Docling payload.

    Returns a dictionary containing:
        total_cells: Total number of cells (including blanks)
        non_blank_cells: Cells with non-empty text
        total_words: Sum of word counts across non-blank cells
        multi_word_cells: Count of cells with more than one word
        ocr_cells: Cells flagged as originating from OCR
        average_words_per_cell: Mean words per non-blank cell
        multi_word_share: Ratio of multi-word cells to non-blank cells
    """
    stats = {
        "total_cells": 0,
        "non_blank_cells": 0,
        "total_words": 0,
        "multi_word_cells": 0,
        "ocr_cells": 0,
        "average_words_per_cell": 0.0,
        "multi_word_share": 0.0,
    }

    parsed_pages = payload.get("parsed_pages", {})
    if not isinstance(parsed_pages, dict):
        return stats

    for page_data in parsed_pages.values():
        cells = page_data.get("cells", [])
        for cell in cells:
            stats["total_cells"] += 1

            if cell.get("from_ocr"):
                stats["ocr_cells"] += 1

            text = cell.get("text", "")
            if not isinstance(text, str):
                continue

            normalized_text = text.strip()
            if not normalized_text:
                continue

            stats["non_blank_cells"] += 1
            words = [word for word in normalized_text.split() if word]
            word_count = len(words)
            stats["total_words"] += word_count

            if word_count > 1:
                stats["multi_word_cells"] += 1

    if stats["non_blank_cells"] > 0:
        stats["average_words_per_cell"] = stats["total_words"] / stats["non_blank_cells"]
        stats["multi_word_share"] = stats["multi_word_cells"] / stats["non_blank_cells"]

    return stats


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    # Use two rows for dynamic programming
    previous_row = list(range(len_b + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """Return similarity score (0-1) based on Levenshtein distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    distance = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return max(0.0, 1.0 - distance / max_len)


def analyze_cells_granularity(cells: List[Dict]) -> Dict[str, Any]:
    """Inspect cells to determine whether content is word-level or phrase-level."""
    stats = {
        "total_cells": len(cells),
        "non_blank_cells": 0,
        "total_words": 0,
        "single_word_cells": 0,
        "average_words_per_cell": 0.0,
        "single_word_share": 0.0,
        "is_word_level": False,
    }

    for cell in cells:
        text = cell.get("text", "")
        if not isinstance(text, str):
            continue
        normalized = text.strip()
        if not normalized:
            continue
        stats["non_blank_cells"] += 1
        words = [w for w in normalized.split() if w]
        word_count = len(words)
        stats["total_words"] += word_count
        if word_count == 1:
            stats["single_word_cells"] += 1

    if stats["non_blank_cells"] > 0:
        stats["average_words_per_cell"] = stats["total_words"] / stats["non_blank_cells"]
        stats["single_word_share"] = stats["single_word_cells"] / stats["non_blank_cells"]
        stats["is_word_level"] = (
            stats["average_words_per_cell"] <= WORD_LEVEL_AVG_THRESHOLD
            and stats["single_word_share"] >= WORD_LEVEL_SINGLE_WORD_SHARE
        )

    return stats


def determine_matching_mode(original_cells: List[Dict], visible_cells: List[Dict]) -> Dict[str, Any]:
    """
    Determine how to compare two cell lists.

    Returns a dict containing:
        - mode: 'word' or 'phrase'
        - original_stats / visible_stats: granularity details
    """
    original_stats = analyze_cells_granularity(original_cells)
    visible_stats = analyze_cells_granularity(visible_cells)

    if original_stats["is_word_level"] and visible_stats["is_word_level"]:
        mode = "word"
    else:
        mode = "phrase"

    return {
        "mode": mode,
        "original_stats": original_stats,
        "visible_stats": visible_stats,
    }


def analyze_flattening_segmentation(
    original_payload: Dict,
    visible_payload: Dict
) -> Dict[str, Any]:
    """
    Determine whether flattening caused a significant segmentation shift that
    will break exact text matching between original and visible payloads.
    """
    original_stats = summarize_payload_cells(original_payload)
    visible_stats = summarize_payload_cells(visible_payload)

    orig_cells = max(1, original_stats["non_blank_cells"])
    vis_cells = max(1, visible_stats["non_blank_cells"])

    cell_count_ratio = vis_cells / orig_cells

    orig_avg_words = original_stats["average_words_per_cell"]
    vis_avg_words = visible_stats["average_words_per_cell"]
    average_word_ratio = (vis_avg_words / orig_avg_words) if orig_avg_words > 0 else 0.0

    multi_word_delta = original_stats["multi_word_share"] - visible_stats["multi_word_share"]

    is_problem_case = (
        cell_count_ratio >= SEGMENTATION_CELL_RATIO_THRESHOLD
        and average_word_ratio <= SEGMENTATION_AVG_WORD_RATIO_THRESHOLD
    )

    analysis = {
        "is_problem_case": is_problem_case,
        "cell_count_ratio": cell_count_ratio,
        "average_word_ratio": average_word_ratio,
        "multi_word_share_delta": multi_word_delta,
        "thresholds": {
            "cell_count_ratio": SEGMENTATION_CELL_RATIO_THRESHOLD,
            "average_word_ratio": SEGMENTATION_AVG_WORD_RATIO_THRESHOLD,
        },
        "original_stats": original_stats,
        "visible_stats": visible_stats,
    }

    return analysis


def find_text_not_visible(
    original_payload: Dict,
    visible_payload: Dict
) -> List[Dict]:
    """
    Find all text that exists in the original payload but NOT in the flattened (visible-only) payload.

    This detects text that is present in the PDF structure but not rendered visually,
    which may indicate hidden text layers (even without bbox overlaps).

    Args:
        original_payload: The original Docling payload
        visible_payload: The flattened (visible-only) Docling payload

    Returns:
        List of dictionaries with information about cells not visible:
        - page: page number
        - cell_index: index in original cells
        - text: the text content
        - bbox: the bounding box
        - font: the font name
        - reason: "not_in_flattened" (text doesn't appear in visible version)
    """
    text_not_visible = []

    if "parsed_pages" not in original_payload or "parsed_pages" not in visible_payload:
        return text_not_visible

    for page_key, page_data in original_payload["parsed_pages"].items():
        original_cells = page_data.get("cells", [])
        visible_cells = visible_payload["parsed_pages"].get(page_key, {}).get("cells", [])
        page_num = page_data.get("page_no", page_key)

        matching_mode = determine_matching_mode(original_cells, visible_cells)["mode"]

        # Check each cell in original
        for idx, orig_cell in enumerate(original_cells):
            orig_text = orig_cell.get('text', '').strip()

            # Skip blank cells
            if not orig_text:
                continue

            # Skip placeholder/unknown text nodes often produced by form fields
            if orig_text.lower() in {"<unknown>", "unknown"}:
                continue

            orig_bbox = orig_cell.get('rect', {})

            # Check if this cell exists in visible payload
            is_visible, _, debug_info = find_cell_in_visible_payload(
                orig_text,
                orig_bbox,
                visible_cells,
                matching_mode
            )

            if not is_visible:
                # This text exists in original but NOT in flattened version
                # Normalize bbox to consistent x0/y0/x1/y1 format
                try:
                    x0, y0, x1, y1 = extract_bbox_coords(orig_bbox)
                    normalized_bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                except (KeyError, TypeError, ValueError):
                    # If bbox extraction fails, use zeros
                    normalized_bbox = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}

                item_record = {
                    "page": page_num,
                    "cell_index": idx,
                    "text": orig_text,
                    "bbox": normalized_bbox,  # Now in consistent x0/y0/x1/y1 format
                    "font": orig_cell.get('font_name', 'Unknown'),
                    "reason": "not_in_flattened"
                }

                if debug_info:
                    item_record["closest_candidate"] = debug_info

                text_not_visible.append(item_record)

    return text_not_visible


def get_pikepdf_payload_path(pdf_path: Path) -> Path:
    """Return the path for storing PikePDF fallback payloads."""
    doc_name = pdf_path.stem
    return PIKEPDF_PAYLOADS_PATH / f"{doc_name}_pikepdf_raw_payload.json"


def get_adobe_payload_path(pdf_path: Path) -> Path:
    """
    Get the path where the Adobe/pikepdf payload should be saved.

    Uses Adobe folder if Adobe API is enabled, otherwise uses pikepdf folder.
    """
    doc_name = pdf_path.stem

    if is_adobe_api_enabled():
        return ADOBE_PAYLOADS_PATH / f"{doc_name}_Adobe_raw_payload.json"
    else:
        return get_pikepdf_payload_path(pdf_path)


def check_adobe_payload_exists(pdf_path: Path) -> Tuple[bool, Optional[Path], Optional[Dict]]:
    """
    Check if an Adobe payload already exists for the PDF.

    Returns:
        Tuple of (exists, payload_path, payload_info)
    """
    payload_path = get_adobe_payload_path(pdf_path)

    if payload_path.exists():
        stats = payload_path.stat()
        payload_info = {
            "size_kb": stats.st_size / 1024,
            "modified": time.ctime(stats.st_mtime)
        }
        return True, payload_path, payload_info
    else:
        return False, payload_path, None


def check_pikepdf_payload_exists(pdf_path: Path) -> Tuple[bool, Optional[Path], Optional[Dict]]:
    """
    Check if a pikepdf payload already exists for the PDF.

    Returns:
        Tuple of (exists, payload_path, payload_info)
    """
    payload_path = get_pikepdf_payload_path(pdf_path)

    if payload_path.exists():
        stats = payload_path.stat()
        payload_info = {
            "size_kb": stats.st_size / 1024,
            "modified": time.ctime(stats.st_mtime)
        }
        return True, payload_path, payload_info
    else:
        return False, payload_path, None


def get_overlapping_bboxes_path(pdf_path: Path) -> Path:
    """Return the path for storing overlapping bboxes detection results."""
    doc_name = pdf_path.stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return OVERLAPPING_BBOXES_PATH / f"{timestamp}_{doc_name}_overlapping_bboxes.json"


def check_overlapping_bboxes_exists(pdf_path: Path) -> Tuple[bool, Optional[Path], Optional[Dict]]:
    """
    Check if overlapping bboxes detection result already exists for the PDF.
    Searches for any existing result with matching PDF name.

    Returns:
        Tuple of (exists, result_path, result_info)
    """
    doc_name = pdf_path.stem

    # Search for existing results (may have timestamp prefix)
    if OVERLAPPING_BBOXES_PATH.exists():
        existing_results = sorted(OVERLAPPING_BBOXES_PATH.glob(f"*{doc_name}_overlapping_bboxes.json"))
        if existing_results:
            # Use the most recent result
            result_path = existing_results[-1]
            stats = result_path.stat()
            result_info = {
                "size_kb": stats.st_size / 1024,
                "modified": time.ctime(stats.st_mtime)
            }
            return True, result_path, result_info

    return False, None, None


def generate_pikepdf_payload(pdf_path: Path, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Path]]:
    """
    Generate Adobe-compatible payload using pikepdf when Adobe API is not available.

    Args:
        pdf_path: Path to the PDF file
        force_regenerate: If True, regenerate even if payload exists

    Returns:
        Tuple of (success, message, payload_path)
    """
    payload_path = get_pikepdf_payload_path(pdf_path)

    # Check if already exists
    if payload_path.exists() and not force_regenerate:
        return True, f"Pikepdf payload already exists (use force regenerate to override)", payload_path

    try:
        # Create payloads directory if it doesn't exist
        PIKEPDF_PAYLOADS_PATH.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Extract font properties using pikepdf
        font_properties = extract_font_properties_with_pikepdf(pdf_path)

        # Extract HasClip information
        hasclip_by_page = analyze_pdf_hasclip(pdf_path)

        # Extract text using PyMuPDF and combine with pikepdf font info
        doc = fitz.open(pdf_path)
        elements = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_index = page_num + 1
            text_dict = page.get_text("dict")
            page_hasclip_events = hasclip_by_page.get(page_index, [])

            span_entries = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # Skip non-text blocks
                    continue

                for line in block.get("lines", []):
                    line_bbox = line.get("bbox", [0, 0, 0, 0])
                    line_height = line_bbox[3] - line_bbox[1] if line_bbox else None
                    for span in line.get("spans", []):
                        text_raw = span.get("text", "")
                        normalized_text = normalize_clip_text(text_raw)
                        if not normalized_text:
                            continue

                        span_entries.append({
                            "text": text_raw.strip(),
                            "font_name": span.get("font", ""),
                            "bbox": span.get("bbox", [0, 0, 0, 0]),
                            "text_size": span.get("size"),
                            "line_height": line_height
                        })

            clip_events = []
            for event in page_hasclip_events or []:
                normalized = normalize_clip_text(event.get("text", ""))
                if normalized or event.get("has_clip"):
                    clip_events.append(bool(event.get("has_clip")))

            if span_entries:
                if clip_events:
                    boundaries = [round(i * len(clip_events) / len(span_entries)) for i in range(len(span_entries) + 1)]
                    clip_per_span = []
                    for idx in range(len(span_entries)):
                        start = boundaries[idx]
                        end = boundaries[idx + 1]
                        if start >= len(clip_events):
                            clip_per_span.append(False)
                            continue
                        slice_flags = clip_events[start:end] or clip_events[start:start + 1]
                        clip_per_span.append(any(slice_flags))
                else:
                    clip_per_span = [False] * len(span_entries)
            else:
                clip_per_span = []

            for idx, span_data in enumerate(span_entries):
                text = span_data["text"]
                font_name_raw = span_data["font_name"]
                bbox = span_data["bbox"]
                text_size = span_data["text_size"]
                line_height = span_data["line_height"]
                has_clip = clip_per_span[idx] if idx < len(clip_per_span) else False

                # Try to match with pikepdf font properties
                # PyMuPDF font names are often like "ABCDEF+FontName"
                font_props = None
                for font_key, props in font_properties.items():
                    if font_name_raw in str(font_key) or str(font_key) in font_name_raw:
                        font_props = props
                        break

                # If no match, create basic font info
                if not font_props:
                    font_props = {
                        'embedded': None,
                        'encoding': None,
                        'font_type': None,
                        'name': font_name_raw
                    }

                element = {
                    "Path": f"//Document/P[{page_index}]/StyleSpan",
                    "Page": page_num,
                    "Text": text,
                    "Font": {
                        "embedded": font_props.get('embedded'),
                        "encoding": font_props.get('encoding'),
                        "family_name": font_props.get('name', '').split('+')[-1] if font_props.get('name') else None,
                        "name": font_props.get('name'),
                        "font_type": font_props.get('font_type'),
                        "monospaced": None,
                        "subset": '+' in font_props.get('name', ''),
                    },
                    "HasClip": has_clip,
                    "TextSize": text_size,
                    "LineHeight": line_height,
                    "BBox": {
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3]
                    }
                }
                elements.append(element)

        doc.close()

        # Create payload structure
        payload = {
            "elements": elements,
            "metadata": {
                "source": "pikepdf",
                "note": "Generated using pikepdf when Adobe API was not available"
            }
        }

        # Save payload
        with open(payload_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        file_size_kb = payload_path.stat().st_size / 1024
        message = f"Pikepdf payload generated in {elapsed_time:.1f}s ({file_size_kb:.1f} KB)"
        return True, message, payload_path

    except Exception as e:
        return False, f"Error generating pikepdf payload: {str(e)}", None


def generate_adobe_payload(pdf_path: Path, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Path]]:
    """
    Generate Adobe raw payload for the PDF, or use pikepdf fallback if Adobe API is not enabled.

    Args:
        pdf_path: Path to the PDF file
        force_regenerate: If True, regenerate even if payload exists

    Returns:
        Tuple of (success, message, payload_path)
    """
    # Check if Adobe API is enabled
    if not is_adobe_api_enabled():
        # Use pikepdf fallback
        return generate_pikepdf_payload(pdf_path, force_regenerate)

    if not HAS_ADOBE:
        return False, "Adobe runner is not available. Check Unit testing - Tools/tools/adobe/", None

    payload_path = get_adobe_payload_path(pdf_path)

    # Check if already exists
    if payload_path.exists() and not force_regenerate:
        return True, f"Adobe payload already exists (use force regenerate to override)", payload_path

    try:
        # Create payloads directory if it doesn't exist
        ADOBE_PAYLOADS_PATH.mkdir(parents=True, exist_ok=True)

        # Run Adobe extraction
        start_time = time.time()
        normalized_result = adobe_run_tool(str(pdf_path))
        elapsed_time = time.time() - start_time

        # Extract raw vendor payload from normalized output
        def _extract_raw_payload(result):
            """Extract raw vendor payload from normalized output"""
            try:
                debug = result.get("debug", {})
                nimt = debug.get("NOT_IN_MASTER_TEMPLATE") or debug.get("not_in_master_template")
                if nimt and isinstance(nimt, dict):
                    # Check for raw_vendor_payload
                    embedded_raw = nimt.get("raw_vendor_payload")
                    if embedded_raw and isinstance(embedded_raw, dict):
                        return embedded_raw

                    # Check for json embedded in raw block
                    raw_block = nimt.get("raw", {})
                    if isinstance(raw_block, dict):
                        embedded_json = raw_block.get("json")
                        if embedded_json and isinstance(embedded_json, dict):
                            return embedded_json

                        # Check for json_path
                        json_path = raw_block.get("json_path") or raw_block.get("vendor_json_path")
                        if json_path and Path(json_path).exists():
                            with open(json_path, 'r', encoding='utf-8') as f:
                                return json.load(f)

                # Fallback to raw_passthrough
                passthrough = debug.get("raw_passthrough", {})
                if isinstance(passthrough, dict):
                    vendor = passthrough.get("vendor", {})
                    if isinstance(vendor, dict):
                        json_path = vendor.get("json_path")
                        if json_path and Path(json_path).exists():
                            with open(json_path, 'r', encoding='utf-8') as f:
                                return json.load(f)

                return None
            except Exception:
                return None

        raw_payload = _extract_raw_payload(normalized_result)

        if raw_payload:
            # Save raw payload to file
            with open(payload_path, 'w', encoding='utf-8') as f:
                json.dump(raw_payload, f, indent=2, ensure_ascii=False)

            file_size_kb = payload_path.stat().st_size / 1024
            message = f"Adobe payload generated in {elapsed_time:.1f}s ({file_size_kb:.1f} KB)"
            return True, message, payload_path
        else:
            # No raw payload found - check if extraction failed
            if normalized_result.get("status") == "error":
                error_msg = normalized_result.get("message", "Adobe extraction failed")
                return False, f"Adobe extraction error: {error_msg}", None
            else:
                return False, "No raw payload found in Adobe output", None

    except Exception as e:
        return False, f"Error generating Adobe payload: {str(e)}", None


def load_adobe_payload_metadata(payload_path: Path) -> Optional[Dict]:
    """
    Load metadata from Adobe payload.

    Returns:
        Dictionary with payload metadata or None if error
    """
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        metadata = {
            "file_size_kb": payload_path.stat().st_size / 1024,
            "top_level_keys": list(payload.keys()) if isinstance(payload, dict) else [],
        }

        # Get Adobe-specific info
        if isinstance(payload, dict):
            # Adobe payloads typically have 'elements' at top level
            if "elements" in payload:
                elements = payload["elements"]
                if isinstance(elements, list):
                    metadata["num_elements"] = len(elements)

                    # Count element types
                    element_types = {}
                    for elem in elements:
                        elem_type = elem.get("Path", "Unknown")
                        # Extract just the last part (e.g., "Text" from "//Document/Text")
                        if "/" in elem_type:
                            elem_type = elem_type.split("/")[-1]
                        element_types[elem_type] = element_types.get(elem_type, 0) + 1

                    metadata["element_types"] = element_types

            # Check for extended metadata
            if "extended_metadata" in payload:
                ext_meta = payload["extended_metadata"]
                if isinstance(ext_meta, dict):
                    metadata["has_extended_metadata"] = True

        return metadata

    except Exception as e:
        return None


def generate_enhanced_text_info(pdf_path: Path, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Path]]:
    """
    Generate enhanced text info with color clustering and edge/gradient analysis.

    This function:
    1. Loads the Docling payload (must exist first)
    2. Matches Docling cells to PyMuPDF spans
    3. For each cell, performs:
       - K-means color clustering (foreground/background separation)
       - Edge/gradient analysis (Sobel gradients + ELA for forgery detection)
    4. Saves complete analysis to JSON

    Args:
        pdf_path: Path to PDF file
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Tuple of (success, message, output_path)
    """
    try:
        # Import backend modules
        from cell_to_span_matching import cell_to_span_matching_pipeline
        from pillow_color_clustering import analyze_foreground_background_colors
        from edge_gradient_analysis import analyze_edge_gradient, create_foreground_mask_from_clustering
        from cell_data_classes import CellColorElement, CellColorAnalysisResult

        # Check output directory
        ENHANCED_TEXT_INFO_PATH.mkdir(exist_ok=True)

        # Construct output path
        pdf_basename = pdf_path.stem
        output_path = ENHANCED_TEXT_INFO_PATH / f"{pdf_basename}_enhanced_text_info.json"

        # Check if already exists
        if output_path.exists() and not force_regenerate:
            return True, f"Enhanced text info already exists: {output_path.name}", output_path

        # Check if Docling payload exists (search by pattern to handle timestamps)
        doc_name = pdf_path.stem
        existing_payloads = sorted(PAYLOADS_PATH.glob(f"*{doc_name}_Docling_raw_payload.json"))

        if not existing_payloads:
            return False, "Docling payload must be generated first", None

        # Use the most recent payload
        docling_payload_path = existing_payloads[-1]

        # Load Docling payload
        with open(docling_payload_path, 'r', encoding='utf-8') as f:
            docling_payload = json.load(f)

        # Check if enhanced payload (has parsed_pages)
        if 'parsed_pages' not in docling_payload:
            return False, "Docling payload missing parsed_pages (regenerate payload)", None

        # Process only page 0 for now (can be extended to all pages)
        page_num = 0

        # Step 1: Match cells to spans
        matching_result = cell_to_span_matching_pipeline(
            payload=docling_payload,
            pdf_path=pdf_path,
            page_num=page_num,
            min_text_similarity=0.85,
            min_bbox_iou=0.3,
            verbose=False
        )

        matches = matching_result['matches']
        match_stats = matching_result['statistics']

        # Step 2: Process each matched cell
        elements = []
        errors = []
        skipped_whitespace = 0

        for idx, match in enumerate(matches):
            cell = match.docling_cell

            # Skip whitespace-only cells (spaces, tabs, newlines, etc.)
            if not cell.text or not cell.text.strip():
                skipped_whitespace += 1
                continue

            try:
                # Step 2a: Color clustering (GPU-first with CPU fallback)
                try:
                    # Try GPU first (faster)
                    color_result = analyze_foreground_background_colors(
                        pdf_path=pdf_path,
                        page_num=page_num,
                        bbox=cell.bbox,
                        dpi=FEATURE_EXTRACTION_DPI,
                        use_gpu=None  # Auto-detect (will use GPU if available)
                    )
                except (RuntimeError, Exception) as gpu_error:
                    # If GPU fails (CUDA error), fall back to CPU
                    if "CUDA" in str(gpu_error) or "cuda" in str(gpu_error):
                        color_result = analyze_foreground_background_colors(
                            pdf_path=pdf_path,
                            page_num=page_num,
                            bbox=cell.bbox,
                            dpi=FEATURE_EXTRACTION_DPI,
                            use_gpu=False  # Force CPU fallback
                        )
                    else:
                        # If not a CUDA error, re-raise
                        raise

                # Step 2b: Create foreground mask from color clustering
                foreground_mask = create_foreground_mask_from_clustering(
                    pdf_path=str(pdf_path),
                    page_num=page_num,
                    bbox=cell.bbox,
                    foreground_color=color_result.foreground_color,
                    background_color=color_result.background_color,
                    dpi=400
                )

                # Step 2c: Edge/gradient analysis
                edge_result = analyze_edge_gradient(
                    pdf_path=str(pdf_path),
                    page_num=page_num,
                    bbox=cell.bbox,
                    foreground_mask=foreground_mask,
                    dpi=400,
                    use_cv2=None,  # Auto-detect
                    ela_quality=95
                )

                # Step 3: Create CellColorElement
                element = CellColorElement(
                    docling_cell=cell,
                    pillow_colors=color_result,
                    edge_gradient_analysis=edge_result,
                    pymupdf_match=match
                )

                elements.append(element)

            except Exception as e:
                # Track errors
                error_msg = f"Cell {idx}: {str(e)}"
                errors.append(error_msg)
                # Show first error in detail for debugging
                if len(errors) == 1:
                    import traceback
                    print(f"\n=== FIRST ERROR DETAILS ===")
                    print(f"Cell text: '{cell.text[:50]}'")
                    print(f"Cell bbox: {cell.bbox}")
                    print(f"Error: {str(e)}")
                    print(f"Traceback:\n{traceback.format_exc()}")
                    print(f"========================\n")
                continue

        # Step 4: Create result container
        result = CellColorAnalysisResult(
            document_name=pdf_path.name,
            page_num=page_num,
            total_cells=match_stats['total_cells'],
            total_spans=match_stats['total_spans'],
            matched_cells=match_stats['matched_cells'],
            match_rate=match_stats['cell_match_rate'],
            elements=elements,
            match_statistics=match_stats
        )

        # Step 5: Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # Include error and skip summary in message
        summary_parts = []
        if skipped_whitespace > 0:
            summary_parts.append(f"Skipped {skipped_whitespace} whitespace")
        if errors:
            summary_parts.append(f"Failed: {len(errors)}")

        if summary_parts:
            summary = f" ({', '.join(summary_parts)})"
        else:
            summary = ""

        success_msg = f"Enhanced text info generated: {output_path.name} ({len(elements)} cells processed){summary}"
        return True, success_msg, output_path

    except ImportError as e:
        return False, f"Missing required module: {str(e)}", None
    except Exception as e:
        return False, f"Error generating enhanced text info: {str(e)}", None


def generate_azure_di_payload(pdf_path: Path, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Path]]:
    """
    Generate Azure DI payload for non-OCRed documents.

    Args:
        pdf_path: Path to PDF file
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Tuple of (success, message, output_path)
    """
    # Check if Azure DI API is enabled
    if not is_azure_di_api_enabled():
        return False, "Azure DI API is not enabled or configured properly", None

    try:
        from azure_document_intelligence import extract_ocr_with_azure_di
        from datetime import datetime

        # Save to unified raw payloads location
        RAW_PAYLOADS_PATH.mkdir(parents=True, exist_ok=True)

        # Construct output path with timestamp
        pdf_basename = pdf_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RAW_PAYLOADS_PATH / f"{timestamp}_{pdf_basename}_Azure_Document_Intelligence_raw_payload.json"

        # Check if recent payload exists (within last hour)
        existing_payloads = sorted(RAW_PAYLOADS_PATH.glob(f"*{pdf_basename}_Azure_Document_Intelligence_raw_payload.json"))
        if existing_payloads and not force_regenerate:
            most_recent = existing_payloads[-1]
            return True, f"Azure DI payload already exists: {most_recent.name}", most_recent

        # Extract with Azure DI
        result = extract_ocr_with_azure_di(
            document_path=str(pdf_path),
            verbose=True,
            timeout=120
        )

        if not result:
            return False, "Azure DI extraction failed (check credentials and API)", None

        # Extract RAW API response (not normalized payload!)
        def _coerce_to_dict(obj):
            if not obj:
                return None
            for attr in ("to_dict", "as_dict", "model_dump"):
                method = getattr(obj, attr, None)
                if callable(method):
                    try:
                        converted = method()
                    except Exception:
                        continue
                    if isinstance(converted, dict):
                        return converted
            generated = getattr(obj, "_to_generated", None)
            if callable(generated):
                try:
                    gen_obj = generated()
                    gen_method = getattr(gen_obj, "to_dict", None)
                    if callable(gen_method):
                        converted = gen_method()
                        if isinstance(converted, dict):
                            return converted
                except Exception:
                    pass
            return None

        raw_result = result.get('raw_result')
        if raw_result and not isinstance(raw_result, dict):
            raw_result = _coerce_to_dict(raw_result)
        if not raw_result:
            raw_result = _coerce_to_dict(result.get('raw_result_object'))

        if raw_result and isinstance(raw_result, dict):
            # Save ONLY the raw API response
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(raw_result, f, indent=2, ensure_ascii=False)
        else:
            # Fallback if no raw result
            return False, "No raw API response available from Azure DI", None

        total_pages = result['metadata']['page_count']
        processing_time = result['metadata']['processing_time']

        success_msg = f"Azure DI RAW payload generated: {output_path.name} ({total_pages} pages, {processing_time:.1f}s)"
        return True, success_msg, output_path

    except ImportError as e:
        return False, f"Azure DI module not available: {str(e)}", None
    except Exception as e:
        return False, f"Error generating Azure DI payload: {str(e)}", None


def generate_enhanced_text_info_from_azure_di(pdf_path: Path, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Path]]:
    """
    Generate enhanced text info from Azure DI payload (same logic as Docling).

    Uses the same color clustering and edge/gradient analysis.
    For non-OCRed documents, uses min(FEATURE_EXTRACTION_DPI, effective_dpi).
    """
    try:
        # Import backend modules
        from pillow_color_clustering import analyze_foreground_background_colors
        from edge_gradient_analysis import analyze_edge_gradient, create_foreground_mask_from_clustering
        from cell_data_classes import CellColorElement, CellColorAnalysisResult, DoclingCell

        # Check output directory
        ENHANCED_TEXT_INFO_PATH.mkdir(exist_ok=True)

        # Calculate effective DPI from source PDF images
        effective_dpi = calculate_effective_dpi(pdf_path)
        if effective_dpi is not None:
            # Use minimum of configured max DPI and effective DPI
            extraction_dpi = int(min(FEATURE_EXTRACTION_DPI, effective_dpi))
        else:
            # Fall back to configured DPI if calculation fails
            extraction_dpi = FEATURE_EXTRACTION_DPI

        # Construct output path
        pdf_basename = pdf_path.stem
        output_path = ENHANCED_TEXT_INFO_PATH / f"{pdf_basename}_Azure_DI_enhanced_text_info.json"

        # Check if already exists
        if output_path.exists() and not force_regenerate:
            return True, f"Azure DI enhanced text info already exists: {output_path.name}", output_path

        # Check if Azure DI payload exists
        azure_di_payload_path = AZURE_DI_PAYLOADS_PATH / f"{pdf_basename}_Azure_DI_payload.json"
        if not azure_di_payload_path.exists():
            return False, "Azure DI payload must be generated first", None

        # Load Azure DI payload
        with open(azure_di_payload_path, 'r', encoding='utf-8') as f:
            azure_di_data = json.load(f)

        normalized_payload = azure_di_data['normalized_payload']

        # Process page 0
        page_num = 0
        if not normalized_payload or len(normalized_payload) == 0:
            return False, "Azure DI payload has no pages", None

        page_data = normalized_payload[page_num]
        cells_data = page_data.get('cells', [])

        if not cells_data:
            return False, "Azure DI payload has no cells on page 0", None

        # Convert Azure DI cells to DoclingCell format
        docling_cells = []
        for cell in cells_data:
            bbox_dict = cell['bbox']
            bbox_tuple = (bbox_dict['x0'], bbox_dict['y0'], bbox_dict['x1'], bbox_dict['y1'])

            docling_cell = DoclingCell(
                text=cell['text'],
                bbox=bbox_tuple,
                font_name=cell.get('font', 'Unknown'),
                confidence=cell.get('confidence'),
                from_ocr=True,
                index=cell.get('cell_id')
            )
            docling_cells.append(docling_cell)

        # Process each cell (same logic as Docling pipeline)
        elements = []
        errors = []
        skipped_whitespace = 0

        for idx, cell in enumerate(docling_cells):
            # Skip whitespace-only cells
            if not cell.text or not cell.text.strip():
                skipped_whitespace += 1
                continue

            try:
                # Color clustering (force CPU for Azure DI to avoid CUDA context issues)
                color_result = analyze_foreground_background_colors(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    bbox=cell.bbox,
                    dpi=extraction_dpi,
                    use_gpu=False  # Force CPU for Azure DI - more stable, avoids CUDA corruption
                )

                # Create foreground mask
                foreground_mask = create_foreground_mask_from_clustering(
                    pdf_path=str(pdf_path),
                    page_num=page_num,
                    bbox=cell.bbox,
                    foreground_color=color_result.foreground_color,
                    background_color=color_result.background_color,
                    dpi=extraction_dpi
                )

                # Edge/gradient analysis
                edge_result = analyze_edge_gradient(
                    pdf_path=str(pdf_path),
                    page_num=page_num,
                    bbox=cell.bbox,
                    foreground_mask=foreground_mask,
                    dpi=extraction_dpi,
                    use_cv2=None,
                    ela_quality=95
                )

                # Create CellColorElement (without PyMuPDF match for Azure DI)
                element = CellColorElement(
                    docling_cell=cell,
                    pillow_colors=color_result,
                    edge_gradient_analysis=edge_result,
                    pymupdf_match=None  # No PyMuPDF matching for Azure DI
                )

                elements.append(element)

            except Exception as e:
                error_msg = f"Cell {idx}: {str(e)}"
                errors.append(error_msg)
                if len(errors) == 1:
                    import traceback
                    print(f"\n=== FIRST ERROR (Azure DI) ===")
                    print(f"Cell text: '{cell.text[:50]}'")
                    print(f"Cell bbox: {cell.bbox}")
                    print(f"Error: {str(e)}")
                    print(f"Traceback:\n{traceback.format_exc()}")
                    print(f"========================\n")
                continue

        # Create result container
        result = CellColorAnalysisResult(
            document_name=pdf_path.name,
            page_num=page_num,
            total_cells=len(cells_data),
            total_spans=0,  # No spans for Azure DI
            matched_cells=len(elements),
            match_rate=100.0 if len(cells_data) > 0 else 0.0,
            elements=elements,
            match_statistics={
                'source': 'azure_di',
                'extraction_dpi': extraction_dpi,
                'effective_dpi': effective_dpi,
                'max_dpi': FEATURE_EXTRACTION_DPI
            }
        )

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # Build summary message
        summary_parts = []
        if skipped_whitespace > 0:
            summary_parts.append(f"Skipped {skipped_whitespace} whitespace")
        if errors:
            summary_parts.append(f"Failed: {len(errors)}")

        if summary_parts:
            summary = f" ({', '.join(summary_parts)})"
        else:
            summary = ""

        success_msg = f"Azure DI enhanced text info generated: {output_path.name} ({len(elements)} cells processed){summary}"
        return True, success_msg, output_path

    except ImportError as e:
        return False, f"Missing required module: {str(e)}", None
    except Exception as e:
        return False, f"Error generating Azure DI enhanced text info: {str(e)}", None


def perform_and_save_cluster_analysis(enhanced_text_info_path: Path, use_gpu: bool = True, max_k: int = 4, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Any]]:
    """
    Perform cluster analysis on enhanced text info and save result.

    Args:
        enhanced_text_info_path: Path to enhanced text info JSON
        use_gpu: Use GPU if available (default: True)
        max_k: Maximum number of clusters for K-means (default: 4)
        force_regenerate: If True, regenerate even if result exists (default: False)

    Returns:
        Tuple of (success, message, cluster_result)
    """
    try:
        if not HAS_CLUSTER_ANALYSIS:
            return False, "Cluster analysis module not available", None

        if not enhanced_text_info_path.exists():
            return False, f"Enhanced text info not found: {enhanced_text_info_path.name}", None

        # Create cluster results directory
        CLUSTER_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

        # Check if result already exists
        result_path = CLUSTER_RESULTS_PATH / f"{enhanced_text_info_path.stem}_cluster_result.pkl"
        if result_path.exists() and not force_regenerate:
            # Load existing result
            import pickle
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            return True, f"Cluster result already exists (use force regenerate to override)", result

        # Perform cluster analysis
        print(f"\n{'='*80}")
        print(f"CLUSTER ANALYSIS")
        print(f"{'='*80}")
        print(f"File: {enhanced_text_info_path.name}")

        result = perform_cluster_analysis(
            enhanced_text_info_path=enhanced_text_info_path,
            use_gpu=use_gpu,
            max_k=max_k,
            verbose=True
        )

        # Save result to pickle file (preserve all data including transformers)
        import pickle
        result_path = CLUSTER_RESULTS_PATH / f"{enhanced_text_info_path.stem}_cluster_result.pkl"
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

        # Also save a summary JSON for easy viewing
        summary_path = CLUSTER_RESULTS_PATH / f"{enhanced_text_info_path.stem}_cluster_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\n✅ Cluster result saved to: {result_path.name}")
        print(f"✅ Summary saved to: {summary_path.name}")

        return True, f"Cluster analysis complete. Results saved to: {result_path.name}", result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error performing cluster analysis: {str(e)}", None


def get_anomaly_explanation(feature_name: str, standardized_value: float) -> str:
    """
    Generate human-readable explanation for an anomalous feature.

    Args:
        feature_name: Name of the feature (e.g., 'fg_luminance')
        standardized_value: Standardized value (number of std devs from mean)

    Returns:
        Human-readable explanation of what to look for
    """
    direction = "much higher" if standardized_value > 0 else "much lower"

    # Feature-specific explanations
    explanations = {
        'fg_luminance': {
            'higher': "Look for text that appears **brighter/lighter** than typical text on this document",
            'lower': "Look for text that appears **darker** than typical text on this document"
        },
        'fg_decimal': {
            'higher': "Text color is **numerically different** (likely different color)",
            'lower': "Text color is **numerically different** (likely different color)"
        },
        'bg_luminance': {
            'higher': "Background behind this text is **brighter/whiter** than typical",
            'lower': "Background behind this text is **darker/grayer** than typical"
        },
        'bg_decimal': {
            'higher': "Background color is **numerically different** (may indicate editing)",
            'lower': "Background color is **numerically different** (may indicate editing)"
        },
        'fg_bg_contrast': {
            'higher': "**Higher contrast** between text and background (text stands out more)",
            'lower': "**Lower contrast** between text and background (text blends more)"
        },
        'top_gradient_mean': {
            'higher': "**Sharper/more distinct** upper edge on text (may indicate digital insertion)",
            'lower': "**Softer/blurrier** upper edge on text"
        },
        'bottom_gradient_mean': {
            'higher': "**Sharper/more distinct** lower edge on text (may indicate digital insertion)",
            'lower': "**Softer/blurrier** lower edge on text"
        },
        'left_gradient_mean': {
            'higher': "**Sharper/more distinct** left edge on text (may indicate digital insertion)",
            'lower': "**Softer/blurrier** left edge on text"
        },
        'right_gradient_mean': {
            'higher': "**Sharper/more distinct** right edge on text (may indicate digital insertion)",
            'lower': "**Softer/blurrier** right edge on text"
        },
        'top_gradient_std': {
            'higher': "**More variation** in upper edge sharpness across characters",
            'lower': "**More uniform** upper edge sharpness"
        },
        'bottom_gradient_std': {
            'higher': "**More variation** in lower edge sharpness across characters",
            'lower': "**More uniform** lower edge sharpness"
        },
        'left_gradient_std': {
            'higher': "**More variation** in left edge sharpness across characters",
            'lower': "**More uniform** left edge sharpness"
        },
        'right_gradient_std': {
            'higher': "**More variation** in right edge sharpness across characters",
            'lower': "**More uniform** right edge sharpness"
        }
    }

    # Get explanation for this feature
    if feature_name in explanations:
        direction_key = 'higher' if standardized_value > 0 else 'lower'
        return explanations[feature_name][direction_key]
    else:
        # Fallback for unknown features
        return f"{direction} than typical"


def detect_hidden_text(pdf_path: Path, payload_path: Path, reuse_visible_payload: bool = False) -> Optional[Dict]:
    """
    Detect hidden text in PDF by comparing original vs visible-only payloads.

    Detection flow:
    1. Load original payload and check for overlapping text bboxes
    2. Always create visible-only PDF (flattened via PDF → Images → PDF pipeline)
    3. Generate payload for visible-only version
    4. Compare payloads to identify hidden text in TWO ways:
       a) Text with overlapping bboxes where one is hidden (not in flattened)
       b) Text that exists in original but not in flattened (hidden without overlap)

    Returns:
        Dictionary with detection results including:
        - hidden_items_with_overlap: Text hidden via overlapping bboxes
        - hidden_items_no_overlap: Text in original but not visible (no overlap)
        - total_hidden: Total count of all hidden text
        - metadata: Comprehensive information about the detection process
    """
    try:
        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[HIDDEN TEXT DETECTION] Starting analysis for: {pdf_path.name}")
        print(f"[START TIME] {time.strftime('%H:%M:%S')}")
        print(f"{'='*80}")

        # Setup paths
        pdf_basename = pdf_path.stem
        visible_pdf = PDF_PROCESSING_PATH / f"{pdf_basename}_ocred.pdf"
        visible_payload_path: Optional[Path] = None

        # STEP 1: Load original payload and check for overlaps FIRST
        # This is much faster than creating visible-only PDF/payload
        step1_start = time.time()
        print(f"\n[STEP 1] Loading original payload and checking for overlaps...")
        print(f"  Original payload: {payload_path.name}")

        load_start = time.time()
        with open(payload_path, 'r', encoding='utf-8') as f:
            original_payload = json.load(f)
        load_elapsed = time.time() - load_start
        print(f"  ⏱️  Payload load time: {load_elapsed:.2f}s")

        # Check if payload has parsed_pages
        if "parsed_pages" not in original_payload:
            print(f"  ✗ Original payload missing 'parsed_pages' key")
            return {"error": "Original payload missing parsed_pages data"}

        orig_pages = len(original_payload["parsed_pages"])
        print(f"  ✓ Original payload: {orig_pages} pages loaded")

        # Quick scan for overlapping cells across all pages
        print(f"  Scanning for overlapping cells...")
        print(f"  IoU threshold: {IOU_THRESHOLD} ({IOU_THRESHOLD*100:.0f}% overlap)")

        scan_start = time.time()
        total_overlaps_found = 0
        total_pages_with_overlaps = 0

        for page_key, page_data in original_payload["parsed_pages"].items():
            original_cells = page_data.get("cells", [])
            page_num = page_data.get("page_no", page_key)

            # Quick overlap check using optimized spatial grid
            page_scan_start = time.time()
            overlapping_pairs = find_overlapping_pairs_optimized(original_cells, threshold=IOU_THRESHOLD)
            page_scan_elapsed = time.time() - page_scan_start

            # Filter out blank-to-blank overlaps
            non_blank_overlaps = []
            for i, j, iou in overlapping_pairs:
                cell1_text = original_cells[i].get('text', '').strip()
                cell2_text = original_cells[j].get('text', '').strip()
                if cell1_text and cell2_text:  # Both non-blank
                    non_blank_overlaps.append((i, j, iou))

            if non_blank_overlaps:
                total_overlaps_found += len(non_blank_overlaps)
                total_pages_with_overlaps += 1
                print(f"    Page {page_num}: {len(non_blank_overlaps)} overlaps found (⏱️ {page_scan_elapsed:.3f}s)")

        scan_elapsed = time.time() - scan_start
        step1_elapsed = time.time() - step1_start
        print(f"\n  ✓ Scan complete: {total_overlaps_found} overlapping cell pairs found across {total_pages_with_overlaps} pages")
        print(f"  ⏱️  Step 1 total time: {step1_elapsed:.2f}s (load: {load_elapsed:.2f}s, scan: {scan_elapsed:.2f}s)")

        # ALWAYS proceed with flattening (no early exit)
        # Even if no overlaps found, we still need to check for text that's not visible
        if total_overlaps_found == 0:
            print(f"\n  ℹ️  No overlapping text found")
            print(f"  → Still proceeding with PDF flattening to detect non-visible text...")
        else:
            print(f"\n  → Overlaps detected! Proceeding with full hidden text detection...")

        # STEP 2: Create visible-only PDF (expensive operation, but necessary)
        step2_start = time.time()
        print(f"\n[STEP 2] Creating visible-only PDF...")
        if reuse_visible_payload:
            if visible_pdf.exists():
                print(f"  ✓ Using existing visible-only PDF: {visible_pdf.name} ({visible_pdf.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"  ✗ Visible-only PDF not found. Please run a full detection first.")
                return {"error": "Visible-only PDF not found. Run full detection before reprocessing."}
        else:
            if not visible_pdf.exists():
                print(f"  Creating visible-only PDF at: {visible_pdf}")
                print(f"  (PDF → Images @ 400 DPI → PDF pipeline)")
                create_visible_only_pdf(pdf_path, visible_pdf)
                print(f"  ✓ Created visible-only PDF ({visible_pdf.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"  ✓ Using existing visible-only PDF: {visible_pdf.name} ({visible_pdf.stat().st_size / 1024:.1f} KB)")
        step2_elapsed = time.time() - step2_start
        print(f"  ⏱️  Step 2 total time: {step2_elapsed:.2f}s")

        # STEP 3: Generate visible-only payload
        step3_start = time.time()
        print(f"\n[STEP 3] Generating visible-only payload...")
        if reuse_visible_payload:
            visible_payload_path = get_latest_visible_payload_path(pdf_path)
            if not visible_payload_path or not visible_payload_path.exists():
                print(f"  ✗ Visible-only payload not found. Please run a full detection first.")
                return {"error": "Visible-only payload not found. Run full detection before reprocessing."}
            print(f"  ✓ Using existing visible-only payload: {visible_payload_path.name}")
        else:
            visible_payload_path = get_payload_path(visible_pdf)
            print(f"  Generating visible-only payload...")
            gen_start = time.time()
            success, message, returned_path = generate_docling_payload(visible_pdf, force_regenerate=True)
            gen_elapsed = time.time() - gen_start
            if not success:
                print(f"  ✗ Failed to generate visible payload: {message}")
                return {"error": f"Failed to generate visible payload: {message}"}
            if returned_path:
                visible_payload_path = returned_path
            print(f"  ✓ Generated visible-only payload: {visible_payload_path.name} (⏱️ {gen_elapsed:.2f}s)")

        # Load visible payload
        print(f"  Loading visible payload...")
        load_vis_start = time.time()
        with open(visible_payload_path, 'r', encoding='utf-8') as f:
            visible_payload = json.load(f)
        load_vis_elapsed = time.time() - load_vis_start

        # Check if visible payload has parsed_pages
        if "parsed_pages" not in visible_payload:
            print(f"  ✗ Visible payload missing 'parsed_pages' key")
            return {"error": "Visible payload missing parsed_pages data"}

        vis_pages = len(visible_payload["parsed_pages"])
        print(f"  ✓ Visible payload: {vis_pages} pages (⏱️ {load_vis_elapsed:.2f}s)")
        step3_elapsed = time.time() - step3_start
        print(f"  ⏱️  Step 3 total time: {step3_elapsed:.2f}s")

        # Analyze segmentation differences introduced by flattening
        segmentation_analysis = analyze_flattening_segmentation(original_payload, visible_payload)
        orig_stats = segmentation_analysis["original_stats"]
        vis_stats = segmentation_analysis["visible_stats"]

        print("\n  Segmentation check (original vs. visible payload):")
        print(f"    Non-blank cells: original={orig_stats['non_blank_cells']} visible={vis_stats['non_blank_cells']} (ratio={segmentation_analysis['cell_count_ratio']:.2f})")
        print(f"    Avg words/cell: original={orig_stats['average_words_per_cell']:.2f} visible={vis_stats['average_words_per_cell']:.2f} (ratio={segmentation_analysis['average_word_ratio']:.2f})")
        print(f"    Multi-word share delta: {segmentation_analysis['multi_word_share_delta']:.3f}")

        if segmentation_analysis["is_problem_case"]:
            print("    -> Detected significant segmentation shift (flagging for advanced matching).")
        else:
            print("    -> Segmentation within expected bounds.")

        # STEP 4: Compare original vs visible to identify hidden text
        step4_start = time.time()
        print(f"\n[STEP 4] Comparing original vs visible payloads to identify hidden text...")
        print(f"  Bbox tolerance: {BBOX_TOLERANCE} ({BBOX_TOLERANCE*100:.0f}% for visibility matching)")

        hidden_text_items = []
        blank_text_locations = []  # Track blank text cells
        blank_cells_zero_bbox = 0
        blank_cells_non_zero_bbox = 0
        total_overlaps = 0
        total_cells_checked = 0
        total_blank_cells_skipped = 0

        for page_key, page_data in original_payload["parsed_pages"].items():
            page_compare_start = time.time()
            original_cells = page_data.get("cells", [])
            visible_cells = visible_payload["parsed_pages"].get(page_key, {}).get("cells", [])

            page_num = page_data.get("page_no", page_key)

            print(f"\n  Page {page_num}:")
            print(f"    Original cells: {len(original_cells)}")
            print(f"    Visible cells:  {len(visible_cells)}")

            matching_info = determine_matching_mode(original_cells, visible_cells)
            matching_mode = matching_info["mode"]
            print(f"    Matching mode: {matching_mode.upper()} (orig avg={matching_info['original_stats']['average_words_per_cell']:.2f}, vis avg={matching_info['visible_stats']['average_words_per_cell']:.2f})")

            # Identify blank cells on this page (unique per cell index)
            blank_cells_by_index: Dict[int, Dict[str, Any]] = {}
            for idx, cell in enumerate(original_cells):
                cell_text_full = cell.get('text', '')
                if not cell_text_full or not cell_text_full.strip():
                    normalized_bbox, is_zero_bbox, bbox_source = normalize_cell_bbox(cell)
                    blank_entry = {
                        "page": page_num,
                        "cell_index": idx,
                        "bbox": normalized_bbox,
                        "font": cell.get('font_name', 'Unknown'),
                        "bbox_is_zero": is_zero_bbox,
                        "bbox_source": bbox_source
                    }
                    blank_cells_by_index[idx] = blank_entry
                    blank_text_locations.append(blank_entry)

                    if is_zero_bbox:
                        blank_cells_zero_bbox += 1
                    else:
                        blank_cells_non_zero_bbox += 1

            page_overlaps = 0
            page_hidden = 0
            page_blank_skipped = len(blank_cells_by_index)
            total_blank_cells_skipped += page_blank_skipped

            # Use optimized spatial grid algorithm to find overlapping cells
            # This reduces complexity from O(n²) to ~O(9cn) where c = avg cells per grid cell
            print(f"    Using spatial grid optimization...")
            overlapping_pairs = find_overlapping_pairs_optimized(original_cells, threshold=IOU_THRESHOLD)

            # Count cells checked (for backwards compatibility with diagnostic output)
            # In optimized mode, we check far fewer pairs but report the actual candidate pairs checked
            n_cells = len(original_cells)
            n_blank = len(blank_cells_by_index)
            n_non_blank = n_cells - n_blank
            naive_checks = (n_non_blank * (n_non_blank - 1)) // 2
            actual_checks = len(overlapping_pairs)
            total_cells_checked += actual_checks

            print(f"    Optimization: {actual_checks} pairs checked (vs {naive_checks} in naive O(n²) approach)")
            print(f"    Speedup: {naive_checks / actual_checks:.1f}x fewer comparisons" if actual_checks > 0 else "    Speedup: N/A (no candidates)")

            # Process each overlapping pair
            for i, j, iou in overlapping_pairs:
                cell1 = original_cells[i]
                cell2 = original_cells[j]

                cell1_text = cell1.get('text', '').strip()
                cell2_text = cell2.get('text', '').strip()

                # Skip if either cell is blank (already recorded uniquely)
                if i in blank_cells_by_index or j in blank_cells_by_index:
                    continue

                total_overlaps += 1
                page_overlaps += 1

                # Debug: Show overlap details
                print(f"      [OVERLAP {page_overlaps}] IoU={iou:.3f}")
                print(f"        Cell {i}: '{cell1_text[:40]}{'...' if len(cell1_text) > 40 else ''}'")
                print(f"        Cell {j}: '{cell2_text[:40]}{'...' if len(cell2_text) > 40 else ''}'")

                # Check visibility
                cell1_visible, cell1_match, _ = find_cell_in_visible_payload(
                    cell1_text, cell1.get('rect', {}), visible_cells, matching_mode
                )
                cell2_visible, cell2_match, _ = find_cell_in_visible_payload(
                    cell2_text, cell2.get('rect', {}), visible_cells, matching_mode
                )

                print(f"        Visibility: Cell {i}={'VISIBLE' if cell1_visible else 'HIDDEN'}, Cell {j}={'VISIBLE' if cell2_visible else 'HIDDEN'}")

                # Determine hidden text
                if cell1_visible and not cell2_visible:
                    # cell2 is hidden
                    page_hidden += 1
                    hidden_text_items.append({
                        "page": page_num,
                        "cell_index": j,
                        "hidden_text": cell2_text,
                        "visible_text": cell1_text,
                        "hidden_bbox": cell2.get('rect', {}),
                        "visible_bbox": cell1.get('rect', {}),
                        "hidden_font": cell2.get('font_name', 'Unknown'),
                        "visible_font": cell1.get('font_name', 'Unknown'),
                        "iou": iou,
                        "confidence": "HIGH"
                    })
                    print(f"        → HIDDEN TEXT DETECTED: '{cell2_text}'")

                elif cell2_visible and not cell1_visible:
                    # cell1 is hidden
                    page_hidden += 1
                    hidden_text_items.append({
                        "page": page_num,
                        "cell_index": i,
                        "hidden_text": cell1_text,
                        "visible_text": cell2_text,
                        "hidden_bbox": cell1.get('rect', {}),
                        "visible_bbox": cell2.get('rect', {}),
                        "hidden_font": cell1.get('font_name', 'Unknown'),
                        "visible_font": cell2.get('font_name', 'Unknown'),
                        "iou": iou,
                        "confidence": "HIGH"
                    })
                    print(f"        → HIDDEN TEXT DETECTED: '{cell1_text}'")
                else:
                    # Both visible or both hidden
                    print(f"        → No forgery (both {'visible' if cell1_visible else 'hidden'})")

            page_compare_elapsed = time.time() - page_compare_start
            print(f"    Summary: {page_overlaps} overlaps, {page_hidden} hidden, {page_blank_skipped} blank cells skipped (⏱️ {page_compare_elapsed:.3f}s)")

        step4_elapsed = time.time() - step4_start
        print(f"\n  ⏱️  Step 4 total time: {step4_elapsed:.2f}s")

        # STEP 5: Find text that's not visible (exists in original but not in flattened)
        # This detects hidden text WITHOUT bbox overlaps (e.g., invisible rendering mode, white-on-white)
        # When flattened via image rendering, hidden text is NOT captured
        step5_start = time.time()
        print(f"\n[STEP 5] Finding text not visible (without bbox overlaps)...")

        text_not_visible = find_text_not_visible(original_payload, visible_payload)

        # Filter out items already detected through overlaps (avoid duplicates)
        # Create set of (page, cell_index) tuples from overlap detections
        overlap_detected_cells = {(item['page'], item['cell_index']) for item in hidden_text_items}

        # Keep only items not already detected
        text_not_visible_filtered = [
            item for item in text_not_visible
            if (item['page'], item['cell_index']) not in overlap_detected_cells
        ]

        step5_elapsed = time.time() - step5_start
        print(f"  ✓ Found {len(text_not_visible)} total cells not in flattened version")
        print(f"  ✓ Filtered to {len(text_not_visible_filtered)} new items (excluding overlap-detected cells)")
        if not SHOW_TEXT_NOT_RENDERED and text_not_visible_filtered:
            print("  ↳ Text-not-visible reporting disabled; filtered items will be suppressed.")
        print(f"  ⏱️  Step 5 total time: {step5_elapsed:.2f}s")

        text_not_visible_output = text_not_visible_filtered if SHOW_TEXT_NOT_RENDERED else []

        # Combine all hidden text
        all_hidden_items = hidden_text_items + text_not_visible_output
        total_hidden_count = len(all_hidden_items)

        # Final summary
        total_elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"[DETECTION SUMMARY]")
        print(f"{'='*80}")
        print(f"  Total cell pairs checked: {total_cells_checked}")
        print(f"  Blank cells skipped: {total_blank_cells_skipped}")
        print(f"  Overlapping cells found: {total_overlaps}")
        print(f"")
        print(f"  HIDDEN TEXT DETECTED:")
        print(f"    With overlap:    {len(hidden_text_items)} items (text hidden via overlapping bboxes)")
        print(f"    Without overlap: {len(text_not_visible_output)} items (invisible rendering, white-on-white, etc.)")
        print(f"    TOTAL HIDDEN:    {total_hidden_count} items")
        print(f"")
        print(f"  ⏱️  TIMING BREAKDOWN:")
        print(f"    Total time:     {total_elapsed:.2f}s (100%)")
        print(f"    Step 1 (scan):  {step1_elapsed:.2f}s ({step1_elapsed/total_elapsed*100:.1f}%)")
        print(f"    Step 2 (PDF):   {step2_elapsed:.2f}s ({step2_elapsed/total_elapsed*100:.1f}%)")
        print(f"    Step 3 (payload): {step3_elapsed:.2f}s ({step3_elapsed/total_elapsed*100:.1f}%)")
        print(f"    Step 4 (overlaps): {step4_elapsed:.2f}s ({step4_elapsed/total_elapsed*100:.1f}%)")
        print(f"    Step 5 (not visible): {step5_elapsed:.2f}s ({step5_elapsed/total_elapsed*100:.1f}%)")
        print(f"{'='*80}\n")

        result = {
            # Separate the two types of hidden text for clarity
            "hidden_items_with_overlap": hidden_text_items,
            "hidden_items_no_overlap": text_not_visible_output,
            "all_hidden_items": all_hidden_items,
            "segmentation_analysis": segmentation_analysis,

            # Summary counts
            "total_hidden": total_hidden_count,
            "total_hidden_with_overlap": len(hidden_text_items),
            "total_hidden_no_overlap": len(text_not_visible_output),
            "total_overlaps": total_overlaps,

            # Legacy field for backwards compatibility (combine all hidden items)
            "hidden_items": all_hidden_items,

            # Other data
            "blank_text_locations": blank_text_locations,
            "total_blank_cells": len(blank_text_locations),
            "visible_pdf": str(visible_pdf) if visible_pdf else None,
            "visible_payload": str(visible_payload_path) if visible_payload_path else None,

            # Comprehensive metadata
            "metadata": {
                "document_name": pdf_path.name,
                "document_path": str(pdf_path),
                "detection_method": "dual_payload_comparison",
                "flattening_performed": True,
                "total_pages_analyzed": orig_pages,
                "pages_with_overlaps": total_pages_with_overlaps,
                "flattening_segmentation_analysis": segmentation_analysis,
            },

            # Debug stats
            "debug_stats": {
                "total_cell_pairs_checked": total_cells_checked,
                "blank_cells_skipped": total_blank_cells_skipped,
                "blank_cells_zero_bbox": blank_cells_zero_bbox,
                "blank_cells_non_zero_bbox": blank_cells_non_zero_bbox,
                "iou_threshold": IOU_THRESHOLD,
                "bbox_tolerance": BBOX_TOLERANCE,
                "text_not_visible_before_filter": len(text_not_visible),
                "text_not_visible_after_filter": len(text_not_visible_output),
            },

            # Timing
            "processing_time": total_elapsed,
            "timing_breakdown": {
                "step1_scan": step1_elapsed,
                "step2_pdf": step2_elapsed,
                "step3_payload": step3_elapsed,
                "step4_overlaps": step4_elapsed,
                "step5_not_visible": step5_elapsed,
            },

            # Metadata
            "pdf_file": str(pdf_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save result to JSON file
        try:
            OVERLAPPING_BBOXES_PATH.mkdir(parents=True, exist_ok=True)
            output_path = get_overlapping_bboxes_path(pdf_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] Detection result saved to: {output_path.name}")
        except Exception as e:
            print(f"[WARNING] Could not save detection result: {e}")

        return result

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Hidden text detection failed:")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# PDF ANNOTATION AND VISUALIZATION
# ============================================================================

BASELINE_THRESHOLD_COUNT = 4
BASELINE_THRESHOLD_PERCENTAGE = 2.0
PER_PAGE_RARE_COUNT = 5


def identify_rare_font_properties(
    analysis: Dict,
    threshold_count: int = BASELINE_THRESHOLD_COUNT,
    threshold_percentage: float = BASELINE_THRESHOLD_PERCENTAGE,
) -> Dict[str, List[Tuple[str, List[Dict]]]]:
    """
    Identify rare font property values from Adobe/pikepdf analysis.

    A property value is considered rare if:
    - It has less than threshold_count items, OR
    - It represents less than threshold_percentage of total items, OR
    - For text size / line height, it appears on any page ≤ PER_PAGE_RARE_COUNT times

    Args:
        analysis: Analysis dictionary from analyze_adobe_payload()
        threshold_count: Minimum count for a value to not be considered rare (default 4)
        threshold_percentage: Minimum percentage for a value to not be considered rare (default 2.0%)

    Returns:
        Dictionary mapping property names to list of (value, items) tuples
    """
    rare_properties = {}

    property_names = ["embedded", "encoding", "family_name", "name", "monospaced", "subset", "has_clip", "text_size", "line_height"]

    for prop_name in property_names:
        if prop_name not in analysis:
            continue

        prop_data = analysis[prop_name]
        total_items = prop_data.get("total", 0)

        if total_items == 0:
            continue

        stats = prop_data.get("stats", {})
        items_dict = prop_data.get("items", {})

        rare_values = []

        for value, count in stats.items():
            percentage = (count / total_items) * 100

            # Check if rare at page-level for target properties
            items = items_dict.get(value, [])
            is_per_page_rare = False

            if prop_name in {"text_size", "line_height"} and items:
                counts_by_page: Dict[int, int] = defaultdict(int)
                for item in items:
                    page_index = item.get("page")
                    if page_index is not None:
                        counts_by_page[int(page_index)] += 1

                if counts_by_page and any(count <= PER_PAGE_RARE_COUNT for count in counts_by_page.values()):
                    is_per_page_rare = True

            # Check if rare
            if count < threshold_count or percentage < threshold_percentage or is_per_page_rare:
                rare_values.append((value, items))

        if rare_values:
            rare_properties[prop_name] = rare_values

    return rare_properties


def render_pdf_with_annotations(
    pdf_path: Path,
    page_num: int,
    hidden_text_items: List[Dict] = None,
    rare_adobe_properties: Dict = None,
    rare_pikepdf_properties: Dict = None,
    docling_table_cells: List[Dict[str, Any]] = None,
    alignment_anomalies: List[Dict[str, Any]] = None,
    alignment_baselines: List[Dict[str, Any]] = None,
    colon_spacing_items: List[Dict[str, Any]] = None,
    dpi: int = 150
) -> Optional[Image.Image]:
    """
    Render a PDF page with bounding box annotations for hidden text, rare properties,
    Docling table structure, alignment anomalies, alignment baselines, and colon spacing patterns.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        hidden_text_items: List of hidden text detection items
        rare_adobe_properties: Rare properties from Adobe analysis
        rare_pikepdf_properties: Rare properties from pikepdf analysis
        docling_table_cells: Table cell bounding boxes extracted from Docling payload
        alignment_anomalies: Detected label/value misalignment items
        alignment_baselines: Detected alignment baselines for visualization
        colon_spacing_items: Detected colon spacing patterns (green/red/orange)
        dpi: DPI for rendering (default 150)

    Returns:
        PIL Image with annotations, or None if error
    """
    if not HAS_PIL:
        return None

    def _sorted_coords(x0_val: float, y0_val: float, x1_val: float, y1_val: float) -> Tuple[float, float, float, float]:
        """
        Ensure coordinates are ordered so that x0 <= x1 and y0 <= y1.
        This prevents downstream drawing helpers from failing when payloads
        provide inverted rectangles (common in mixed coordinate systems).
        """
        min_x, max_x = sorted((float(x0_val), float(x1_val)))
        min_y, max_y = sorted((float(y0_val), float(y1_val)))
        return min_x, min_y, max_x, max_y

    try:
        # Convert PDF page to image
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_num+1, last_page=page_num+1)
        if not images:
            return None

        img = images[0]
        draw = ImageDraw.Draw(img, 'RGBA')

        # Color scheme for different anomaly types
        colors = {
            'hidden_text': (255, 0, 0, 80),  # Red - Hidden text (visible layer)
            'embedded': (255, 165, 0, 60),   # Orange
            'encoding': (255, 255, 0, 60),   # Yellow
            'family_name': (0, 255, 0, 60),  # Green
            'name': (0, 255, 255, 60),       # Cyan
            'monospaced': (0, 0, 255, 60),   # Blue
            'subset': (128, 0, 128, 60),     # Purple
            'has_clip': (255, 192, 203, 60), # Pink
            'text_size': (255, 140, 0, 60),  # Dark Orange
            'line_height': (139, 69, 19, 60), # Brown
            'docling_alignment': (255, 20, 147, 90)  # Deep pink - alignment anomalies
        }
        hidden_underlay_outline = (30, 144, 255, 255)  # Dodger blue
        hidden_underlay_fill = (30, 144, 255, 60)

        # Scale factor (PDF points to image pixels)
        pdf_doc = fitz.open(pdf_path)
        pdf_page = pdf_doc[page_num]
        pdf_rect = pdf_page.rect
        pdf_width = pdf_rect.width
        pdf_height = pdf_rect.height

        img_width, img_height = img.size
        scale_x = img_width / pdf_width
        scale_y = img_height / pdf_height

        pdf_doc.close()

        annotations = []  # Track what we've drawn

        # Draw hidden text bounding boxes
        if hidden_text_items:
            for item in hidden_text_items:
                if item.get('page') != page_num:
                    continue

                # Get bbox - handle both item types:
                # - Items with overlap have 'visible_bbox' (the visible layer)
                # - Items without overlap have 'bbox' (the non-rendered text)
                bbox = item.get('visible_bbox', item.get('bbox', {}))

                # Extract bbox coordinates (handle both formats)
                try:
                    if 'r_x0' in bbox:
                        # Rotated rectangle format
                        x_coords = [bbox.get('r_x0', 0), bbox.get('r_x1', 0),
                                   bbox.get('r_x2', 0), bbox.get('r_x3', 0)]
                        y_coords = [bbox.get('r_y0', 0), bbox.get('r_y1', 0),
                                   bbox.get('r_y2', 0), bbox.get('r_y3', 0)]
                        x0, x1 = min(x_coords), max(x_coords)
                        y0, y1 = min(y_coords), max(y_coords)
                    elif 'x0' in bbox:
                        # Simple format
                        x0, y0, x1, y1 = _sorted_coords(
                            bbox.get('x0', 0),
                            bbox.get('y0', 0),
                            bbox.get('x1', 0),
                            bbox.get('y1', 0)
                        )
                    else:
                        continue

                    # Scale to image coordinates
                    px0 = int(x0 * scale_x)
                    py0 = int(y0 * scale_y)
                    px1 = int(x1 * scale_x)
                    py1 = int(y1 * scale_y)

                    # Draw filled rectangle
                    draw.rectangle([px0, py0, px1, py1], fill=colors['hidden_text'], outline=(255, 0, 0, 255), width=2)
                    annotations.append('hidden_text')

                except Exception:
                    continue

                hidden_bbox = item.get('hidden_bbox', {})
                if hidden_bbox:
                    try:
                        hx0, hy0, hx1, hy1 = extract_bbox_coords(hidden_bbox)
                        phx0 = int(hx0 * scale_x)
                        phy0 = int(hy0 * scale_y)
                        phx1 = int(hx1 * scale_x)
                        phy1 = int(hy1 * scale_y)

                        draw.rectangle([phx0, phy0, phx1, phy1], outline=hidden_underlay_outline, fill=hidden_underlay_fill, width=2)
                    except Exception:
                        pass

        # Helper function to draw bounding boxes for rare properties
        def draw_rare_property_bboxes(rare_props: Dict, source: str):
            for prop_name, rare_values in rare_props.items():
                if prop_name not in colors:
                    continue

                color = colors[prop_name]

                for value, items in rare_values:
                    for item in items:
                        if item.get('page') != page_num:
                            continue

                        # Get bbox - Adobe/pikepdf payloads have BBox key
                        bbox = item.get('BBox')
                        if not bbox:
                            continue

                        try:
                            x0, y0, x1, y1 = _sorted_coords(
                                bbox.get('x0', 0),
                                bbox.get('y0', 0),
                                bbox.get('x1', 0),
                                bbox.get('y1', 0)
                            )

                            # Scale to image coordinates
                            px0 = int(x0 * scale_x)
                            py0 = int(y0 * scale_y)
                            px1 = int(x1 * scale_x)
                            py1 = int(y1 * scale_y)

                            # Draw filled rectangle
                            outline_color = tuple(list(color[:3]) + [255])
                            draw.rectangle([px0, py0, px1, py1], fill=color, outline=outline_color, width=2)

                            if prop_name not in annotations:
                                annotations.append(prop_name)

                        except Exception:
                            continue

        # Draw rare Adobe properties
        if rare_adobe_properties:
            draw_rare_property_bboxes(rare_adobe_properties, 'Adobe')

        # Draw rare pikepdf properties
        if rare_pikepdf_properties:
            draw_rare_property_bboxes(rare_pikepdf_properties, 'pikepdf')

        # Draw Docling table outlines (light green)
        if docling_table_cells:
            table_outline = (144, 238, 144, 255)
            for cell in docling_table_cells:
                if cell.get('page') != page_num:
                    continue

                bbox = cell.get('bbox', {})
                if not bbox:
                    continue

                try:
                    x0, y0, x1, y1 = _sorted_coords(
                        bbox.get('x0', 0),
                        bbox.get('y0', 0),
                        bbox.get('x1', 0),
                        bbox.get('y1', 0)
                    )
                except (TypeError, ValueError):
                    continue

                px0 = int(x0 * scale_x)
                py0 = int(y0 * scale_y)
                px1 = int(x1 * scale_x)
                py1 = int(y1 * scale_y)

                draw.rectangle([px0, py0, px1, py1], outline=table_outline, width=1)

        # Draw Docling alignment anomalies
        if alignment_anomalies:
            fill_color = colors['docling_alignment']
            outline_color = (fill_color[0], fill_color[1], fill_color[2], 255)

            for item in alignment_anomalies:
                if item.get('page') != page_num:
                    continue

                bbox = item.get('bbox', {})
                if not bbox:
                    continue

                try:
                    x0, y0, x1, y1 = _sorted_coords(
                        bbox.get('x0', 0),
                        bbox.get('y0', 0),
                        bbox.get('x1', 0),
                        bbox.get('y1', 0)
                    )
                except (TypeError, ValueError):
                    continue

                px0 = int(x0 * scale_x)
                py0 = int(y0 * scale_y)
                px1 = int(x1 * scale_x)
                py1 = int(y1 * scale_y)

                draw.rectangle([px0, py0, px1, py1], fill=fill_color, outline=outline_color, width=3)
                annotations.append('docling_alignment')

        # Draw alignment baselines
        if alignment_baselines:
            # Try to load a font for labels, fall back to default if unavailable
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = None

            for baseline in alignment_baselines:
                orientation = baseline.get('orientation')
                value = baseline.get('value')
                page_position = baseline.get('page_position')
                count = baseline.get('count', 0)

                if value is None:
                    continue

                # Choose color based on orientation and page position
                if orientation == 'left':
                    # Left-aligned baselines: cyan/blue shades
                    if page_position == 'page_left':
                        line_color = (0, 191, 255, 255)  # Deep sky blue - solid for page edges
                    else:
                        line_color = (0, 255, 255, 200)  # Cyan - semi-transparent for mid-page
                elif orientation == 'right':
                    # Right-aligned baselines: orange/yellow shades
                    if page_position == 'page_right':
                        line_color = (255, 140, 0, 255)  # Dark orange - solid for page edges
                    else:
                        line_color = (255, 165, 0, 200)  # Orange - semi-transparent for mid-page
                else:
                    continue

                # Convert PDF x-coordinate to pixel coordinate
                px = int(value * scale_x)

                # Draw vertical line across the entire page height
                draw.line([(px, 0), (px, img_height)], fill=line_color, width=2)

                # Add label at the top of the page
                label_text = f"{orientation[0].upper()}: {value:.1f}pt ({count})"

                # Position label slightly offset from the line
                label_x = px + 5 if orientation == 'left' else px - 70
                label_y = 10

                # Draw label background for readability
                if font:
                    bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                else:
                    # Estimate bbox without font
                    bbox = (label_x, label_y, label_x + len(label_text) * 7, label_y + 15)

                bg_color = (255, 255, 255, 200)  # Semi-transparent white
                draw.rectangle(bbox, fill=bg_color)

                # Draw the text
                text_color = (0, 0, 0, 255)  # Black
                if font:
                    draw.text((label_x, label_y), label_text, fill=text_color, font=font)
                else:
                    draw.text((label_x, label_y), label_text, fill=text_color)

                annotations.append('baseline')

        # Draw colon spacing patterns (green/red/orange highlighting)
        if colon_spacing_items:
            for item in colon_spacing_items:
                if item.get('page') != page_num:
                    continue

                classification = item.get('classification')
                bbox = item.get('bbox', {})

                if not bbox:
                    continue

                try:
                    x0, y0, x1, y1 = _sorted_coords(
                        bbox.get('x0', 0),
                        bbox.get('y0', 0),
                        bbox.get('x1', 0),
                        bbox.get('y1', 0)
                    )
                except (TypeError, ValueError):
                    continue

                px0 = int(x0 * scale_x)
                py0 = int(y0 * scale_y)
                px1 = int(x1 * scale_x)
                py1 = int(y1 * scale_y)

                # Color based on classification
                if classification == 'consistent':
                    # Green - matches dominant pattern
                    fill_color = (0, 255, 0, 50)  # Semi-transparent green
                    outline_color = (0, 200, 0, 255)  # Solid green
                    outline_width = 1
                elif classification == 'deviation':
                    # Red - spacing deviation (potential forgery)
                    fill_color = (255, 0, 0, 70)  # More opaque red
                    outline_color = (255, 0, 0, 255)  # Solid red
                    outline_width = 3
                elif classification == 'right_aligned':
                    # Orange - right-aligned exception
                    fill_color = (255, 165, 0, 60)  # Semi-transparent orange
                    outline_color = (255, 140, 0, 255)  # Solid dark orange
                    outline_width = 2
                else:
                    continue

                # Draw rectangle
                draw.rectangle([px0, py0, px1, py1], fill=fill_color, outline=outline_color, width=outline_width)
                annotations.append(f'colon_spacing_{classification}')

        return img

    except Exception as e:
        st.error(f"Error rendering PDF with annotations: {e}")
        return None


def build_annotation_details(
    page_num: int,
    hidden_text_items: List[Dict] = None,
    rare_adobe_properties: Dict = None,
    rare_pikepdf_properties: Dict = None,
    adobe_analysis: Dict = None,
    pikepdf_analysis: Dict = None,
    docling_alignment_items: List[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Build detailed information about all annotated items on a page.

    Returns:
        List of annotation details with text, color, and all reasons
    """
    color_map = {
        'hidden_text': ('#FF0000', 'Red'),
        'embedded': ('#FFA500', 'Orange'),
        'encoding': ('#FFFF00', 'Yellow'),
        'family_name': ('#00FF00', 'Green'),
        'name': ('#00FFFF', 'Cyan'),
        'monospaced': ('#0000FF', 'Blue'),
        'subset': ('#800080', 'Purple'),
        'has_clip': ('#FFC0CB', 'Pink'),
        'text_size': ('#FF8C00', 'Dark Orange'),
        'line_height': ('#8B4513', 'Brown'),
        'docling_alignment': ('#FF1493', 'Deep Pink')
    }

    # Collect all annotated items by text+bbox
    annotations_map = {}  # key: (text, bbox_str) -> {reasons: [], color: str}

    # Add hidden text items
    if hidden_text_items:
        for item in hidden_text_items:
            if item.get('page') != page_num:
                continue

            # Handle both item types:
            # - Items with overlap have 'visible_text' and 'visible_bbox'
            # - Items without overlap have 'text' and 'bbox'
            text = item.get('visible_text', item.get('text', ''))
            bbox = item.get('visible_bbox', item.get('bbox', {}))
            bbox_key = str(bbox)

            key = (text, bbox_key)
            if key not in annotations_map:
                annotations_map[key] = {
                    'text': text,
                    'reasons': [],
                    'color': color_map['hidden_text'][0],
                    'color_name': color_map['hidden_text'][1]
                }

            # Add reason based on item type
            if 'iou' in item:
                # Item with overlap
                iou = item.get('iou', 0)
                iou_percent = iou * 100
                annotations_map[key]['reasons'].append({
                    'type': 'Overlapping bbox',
                    'details': f"{iou_percent:.1f}% overlap",
                    'hidden_text': item.get('hidden_text', '')
                })
            else:
                # Item without overlap (not visible in flattened version)
                annotations_map[key]['reasons'].append({
                    'type': 'Text not rendered visually',
                    'details': 'Exists in PDF structure but not in flattened version',
                    'hidden_text': text
                })

    # Add rare property items
    def add_rare_property_items(rare_props: Dict, source: str, analysis: Dict):
        if not rare_props or not analysis:
            return

        for prop_name, rare_values in rare_props.items():
            if prop_name not in color_map:
                continue

            for value, items in rare_values:
                for item in items:
                    if item.get('page') != page_num:
                        continue

                    text = item.get('text', '')
                    bbox = item.get('BBox')
                    if not bbox:
                        continue

                    bbox_key = str(bbox)
                    key = (text, bbox_key)

                    if key not in annotations_map:
                        annotations_map[key] = {
                            'text': text,
                            'reasons': [],
                            'color': color_map[prop_name][0],
                            'color_name': color_map[prop_name][1]
                        }

                    # Calculate percentage
                    prop_data = analysis.get(prop_name, {})
                    total = prop_data.get('total', 0)
                    stats = prop_data.get('stats', {})
                    count = stats.get(value, 0)
                    percentage = (count / total * 100) if total > 0 else 0

                    display_value = value
                    if prop_name in ("text_size", "line_height"):
                        display_value = format_rounding_classification(value)

                    # Format property name
                    prop_display = {
                        'embedded': 'Embedded',
                        'encoding': 'Encoding',
                        'family_name': 'Family Name',
                        'name': 'Font Name',
                        'monospaced': 'Monospaced',
                        'subset': 'Subset',
                        'has_clip': 'HasClip',
                        'text_size': 'Text Size',
                        'line_height': 'Line Height'
                    }.get(prop_name, prop_name)

                    details_text = f"{display_value} (Rare {percentage:.1f}%)"
                    if prop_name == "text_size" and "exact_size" in item:
                        details_text += f" – Exact {item['exact_size']}"
                    if prop_name == "line_height" and "exact_height" in item:
                        details_text += f" – Exact {item['exact_height']}"

                    annotations_map[key]['reasons'].append({
                        'type': prop_display,
                        'details': details_text,
                        'count': count,
                        'total': total
                    })

                    # Include exact measurement context for size/height
                    if prop_name == "text_size" and "exact_size" in item:
                        annotations_map[key]['reasons'][-1]['details'] = f"{display_value} (Rare {percentage:.1f}%) – Exact {item['exact_size']}"
                    if prop_name == "line_height" and "exact_height" in item:
                        annotations_map[key]['reasons'][-1]['details'] = f"{display_value} (Rare {percentage:.1f}%) – Exact {item['exact_height']}"

    # Add rare Adobe properties
    if rare_adobe_properties and adobe_analysis:
        add_rare_property_items(rare_adobe_properties, 'Adobe', adobe_analysis)

    # Add rare pikepdf properties
    if rare_pikepdf_properties and pikepdf_analysis:
        add_rare_property_items(rare_pikepdf_properties, 'pikepdf', pikepdf_analysis)

    # Add Docling alignment anomalies
    if docling_alignment_items:
        for item in docling_alignment_items:
            if item.get('page') != page_num:
                continue

            text = item.get('value_text', item.get('text', ''))
            bbox = item.get('bbox', {})
            if not bbox:
                continue

            bbox_key = str(bbox)
            key = (text, bbox_key)

            if key not in annotations_map:
                annotations_map[key] = {
                    'text': text,
                    'reasons': [],
                    'color': color_map['docling_alignment'][0],
                    'color_name': color_map['docling_alignment'][1]
                }

            for reason in item.get('reasons', []):
                annotations_map[key]['reasons'].append({
                    'type': reason.get('type', 'Alignment anomaly'),
                    'details': reason.get('details', '')
                })

    # Convert to list and sort by text
    annotations_list = list(annotations_map.values())
    annotations_list.sort(key=lambda x: x['text'])

    return annotations_list


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PDF OCR Detector",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 PDF OCR Detection Tool")
    st.markdown("---")

    # Check if test samples directory exists
    if not TEST_SAMPLES_PATH.exists():
        st.error(f"Test samples directory not found: {TEST_SAMPLES_PATH}")
        st.info(f"Expected path: {TEST_SAMPLES_PATH.absolute()}")
        return

    # Get list of PDF files
    pdf_files = get_pdf_files()

    if not pdf_files:
        st.warning("No PDF files found in the test samples directory.")
        return

    st.success(f"Found {len(pdf_files)} PDF files in test samples directory")

    # File selection
    st.subheader("Select a PDF File")
    selected_file = st.selectbox(
        "Choose a PDF to analyze:",
        pdf_files,
        index=0
    )

    if selected_file:
        pdf_path = TEST_SAMPLES_PATH / selected_file

        st.markdown("---")
        st.subheader(f"Analysis Results: {selected_file}")

        # Create columns for results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### OCR Status")
            with st.spinner("Checking for searchable text..."):
                has_text, total_pages, pages_with_text, text_preview = check_pdf_has_text(pdf_path)

            if has_text:
                st.success("✅ This PDF has been OCRed (contains searchable text)")
                st.metric("Total Pages", total_pages)
                st.metric("Pages with Text", pages_with_text)

                # Calculate percentage
                percentage = (pages_with_text / total_pages * 100) if total_pages > 0 else 0
                st.progress(percentage / 100)
                st.caption(f"{percentage:.1f}% of pages contain text")

                # Show text preview
                if text_preview:
                    with st.expander("📝 Text Preview (First page with text)"):
                        first_page = min(text_preview.keys())
                        st.text(f"Page {first_page}:")
                        st.text(text_preview[first_page])
            else:
                st.warning("⚠️ This PDF has NOT been OCRed (scanned document with no searchable text)")
                st.metric("Total Pages", total_pages)
                st.info("This appears to be a scanned document without OCR processing.")

        with col2:
            st.markdown("### Security Restrictions")
            with st.spinner("Checking for restrictions..."):
                restrictions = check_pdf_restrictions(pdf_path)

            if restrictions.get("is_encrypted"):
                st.warning("🔒 This PDF is encrypted")

                if restrictions.get("needs_password"):
                    st.error("❌ Password required to open this document")
                else:
                    st.info("ℹ️ Document is encrypted but can be opened without password")

                if restrictions.get("has_restrictions"):
                    st.warning("⚠️ Document has security restrictions")
                    st.caption(restrictions.get("restriction_note", ""))

                    # Show detailed permissions if available
                    if HAS_PIKEPDF:
                        with st.expander("🔍 Detailed Permissions"):
                            perm_col1, perm_col2 = st.columns(2)
                            with perm_col1:
                                st.write("**Printing:**", "✅ Allowed" if restrictions.get("can_print") else "❌ Restricted")
                                st.write("**Copying Text:**", "✅ Allowed" if restrictions.get("can_copy") else "❌ Restricted")
                            with perm_col2:
                                st.write("**Modifying:**", "✅ Allowed" if restrictions.get("can_modify") else "❌ Restricted")
                                st.write("**Annotations:**", "✅ Allowed" if restrictions.get("can_annotate") else "❌ Restricted")
            else:
                st.success("✅ No encryption or restrictions detected")
                st.info("This PDF is not encrypted and has no access restrictions.")

                # Show all permissions as allowed
                if HAS_PIKEPDF:
                    with st.expander("🔍 Permissions (All Allowed)"):
                        perm_col1, perm_col2 = st.columns(2)
                        with perm_col1:
                            st.write("**Printing:**", "✅ Allowed")
                            st.write("**Copying Text:**", "✅ Allowed")
                        with perm_col2:
                            st.write("**Modifying:**", "✅ Allowed")
                            st.write("**Annotations:**", "✅ Allowed")

        # Additional file information
        st.markdown("---")
        st.subheader("File Information")
        file_stats = pdf_path.stat()

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("File Size", f"{file_stats.st_size / 1024 / 1024:.2f} MB")
        with info_col2:
            st.metric("File Path", "...")
            st.caption(str(pdf_path.absolute()))
        with info_col3:
            try:
                doc = fitz.open(pdf_path)
                st.metric("PDF Version", f"PDF {doc.metadata.get('format', 'Unknown')}")
                doc.close()
            except:
                st.metric("PDF Version", "Unknown")

        # Docling Payload Generation Section (only for OCRed documents)
        if has_text:
            st.markdown("---")
            st.subheader("🔧 Docling Payload Generation")

            if not HAS_DOCLING:
                st.warning("⚠️ Docling is not installed. Install with: `pip install docling`")
            else:
                # Check if payload exists
                payload_exists, payload_path, payload_info = check_payload_exists(pdf_path)

                payload_col1, payload_col2 = st.columns([2, 1])

                with payload_col1:
                    if payload_exists:
                        st.success(f"✅ Docling payload exists: `{payload_path.name}`")

                        # Load and display metadata
                        metadata = load_payload_metadata(payload_path)
                        if metadata:
                            # Show word-level indicator prominently
                            if metadata.get("has_word_level"):
                                st.success("✅ Enhanced payload with word-level detail & font info")
                            else:
                                st.info("ℹ️ Standard payload (regenerate for word-level detail)")

                            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                            with meta_col1:
                                st.metric("Payload Size", f"{metadata['file_size_kb']:.1f} KB")
                            with meta_col2:
                                if "num_pages" in metadata:
                                    st.metric("Pages", metadata["num_pages"])
                            with meta_col3:
                                if "num_cells" in metadata and metadata["num_cells"] != "N/A":
                                    st.metric("Word Cells", metadata["num_cells"])
                                elif "num_text_elements" in metadata:
                                    st.metric("Text Elements", metadata["num_text_elements"])
                            with meta_col4:
                                if "num_tables" in metadata and metadata["num_tables"] != "N/A":
                                    st.metric("Tables", metadata["num_tables"])

                            # Show additional info in expander
                            with st.expander("📊 Payload Details"):
                                st.write("**Top-level keys:**", ", ".join(metadata["top_level_keys"]))
                                st.write("**Last Modified:**", payload_info["modified"])
                                st.write("**File Path:**", str(payload_path.absolute()))

                                if metadata.get("has_word_level"):
                                    st.write("**Word-level data:** ✅ Included")
                                    st.write("**Font information:** ✅ Included")
                                    st.write("**Color information:** ✅ Included")
                                else:
                                    st.write("**Word-level data:** ❌ Not included (old format)")

                                if "generation_time" in metadata and metadata["generation_time"] != "N/A":
                                    st.write(f"**Generation Time:** {metadata['generation_time']:.1f}s")
                    else:
                        st.info("ℹ️ No Docling payload exists for this document yet")
                        st.caption(f"Payload will be saved to: `{payload_path.name}`")
                        st.caption(f"Folder: `{payload_path.parent.absolute()}`")

                with payload_col2:
                    # Generate Docling button
                    if payload_exists:
                        button_label = "🔄 Regenerate Docling Payload"
                        force_regen = True
                    else:
                        button_label = "⚡ Generate Docling Payload"
                        force_regen = False

                    if st.button(button_label, use_container_width=True, key="generate_docling"):
                        with st.spinner("Generating Docling payload... This may take a minute."):
                            success, message, result_path = generate_docling_payload(
                                pdf_path,
                                force_regenerate=force_regen
                            )

                            if success:
                                st.success(message)
                                st.rerun()  # Refresh to show new payload info
                            else:
                                st.error(message)

                    # Open folder button
                    if st.button("📁 Open Docling Payloads Folder", use_container_width=True, key="open_docling_folder"):
                        # Create folder if it doesn't exist
                        PAYLOADS_PATH.mkdir(parents=True, exist_ok=True)
                        # Open in file explorer (Windows)
                        try:
                            os.startfile(str(PAYLOADS_PATH.absolute()))
                        except:
                            st.info(f"Payloads folder: {PAYLOADS_PATH.absolute()}")

                # Adobe/pikepdf Payload Generation Section (only for OCRed documents)
                st.markdown("---")

                # Dynamic heading based on which method is being used
                using_pikepdf = not is_adobe_api_enabled()
                if using_pikepdf:
                    st.subheader("🎨 pikepdf Payload Generation")
                else:
                    st.subheader("🎨 Adobe Payload Generation")

                # Show info about which method will be used
                if using_pikepdf:
                    if not USE_ADOBE_API_IF_AVAILABLE:
                        st.info("ℹ️ Adobe API is disabled (USE_ADOBE_API_IF_AVAILABLE=False). Using pikepdf fallback.")
                    elif not HAS_ADOBE:
                        st.info("ℹ️ Adobe runner is not available. Using pikepdf fallback for font extraction.")
                    else:
                        st.info("ℹ️ Adobe API requirements not met. Using pikepdf fallback.")
                else:
                    st.info("✅ Adobe API is enabled and available.")

                # Show payload status regardless of Adobe availability (pikepdf can generate too)
                if True:
                    # Check if Adobe/pikepdf payload exists
                    adobe_exists, adobe_path, adobe_info = check_adobe_payload_exists(pdf_path)
                    payload_type = "pikepdf" if using_pikepdf else "Adobe"

                    adobe_col1, adobe_col2 = st.columns([2, 1])

                    with adobe_col1:
                        if adobe_exists:
                            st.success(f"✅ {payload_type} payload exists: `{adobe_path.name}`")

                            # Load and display metadata
                            adobe_metadata = load_adobe_payload_metadata(adobe_path)
                            if adobe_metadata:
                                meta_col1, meta_col2, meta_col3 = st.columns(3)
                                with meta_col1:
                                    st.metric("Payload Size", f"{adobe_metadata['file_size_kb']:.1f} KB")
                                with meta_col2:
                                    if "num_elements" in adobe_metadata:
                                        st.metric("Elements", adobe_metadata["num_elements"])
                                with meta_col3:
                                    if "element_types" in adobe_metadata:
                                        st.metric("Element Types", len(adobe_metadata["element_types"]))

                                # Show additional info in expander
                                with st.expander(f"📊 {payload_type} Payload Details"):
                                    st.write("**Top-level keys:**", ", ".join(adobe_metadata["top_level_keys"]))
                                    st.write("**Last Modified:**", adobe_info["modified"])
                                    st.write("**Folder:**", str(adobe_path.parent.absolute()))
                                    st.write("**File Path:**", str(adobe_path.absolute()))

                                    if "element_types" in adobe_metadata:
                                        st.write("**Element Types:**")
                                        for elem_type, count in adobe_metadata["element_types"].items():
                                            st.write(f"  - {elem_type}: {count}")

                                    if adobe_metadata.get("has_extended_metadata"):
                                        st.write("**Extended Metadata:** ✅ Included")
                        else:
                            st.info(f"ℹ️ No {payload_type} payload exists for this document yet")
                            st.caption(f"Payload will be saved to: `{adobe_path.name}`")
                            st.caption(f"Folder: `{adobe_path.parent.absolute()}`")

                    with adobe_col2:
                        # Generate payload button (Adobe or pikepdf)
                        if adobe_exists:
                            adobe_button_label = f"🔄 Regenerate {payload_type} Payload"
                            adobe_force_regen = True
                        else:
                            adobe_button_label = f"⚡ Generate {payload_type} Payload"
                            adobe_force_regen = False

                        if st.button(adobe_button_label, use_container_width=True, key="generate_adobe"):
                            with st.spinner(f"Generating {payload_type} payload... This may take a minute."):
                                success, message, result_path = generate_adobe_payload(
                                    pdf_path,
                                    force_regenerate=adobe_force_regen
                                )

                                if success:
                                    st.success(message)
                                    st.rerun()  # Refresh to show new payload info
                                else:
                                    st.error(message)

                        # Open folder button (Adobe or pikepdf)
                        folder_path = PIKEPDF_PAYLOADS_PATH if using_pikepdf else ADOBE_PAYLOADS_PATH
                        folder_name = "pikepdf" if using_pikepdf else "Adobe"

                        if st.button(f"📁 Open {folder_name} Payloads Folder", use_container_width=True, key="open_adobe_folder"):
                            # Create folder if it doesn't exist
                            folder_path.mkdir(parents=True, exist_ok=True)
                            # Open in file explorer (Windows)
                            try:
                                os.startfile(str(folder_path.absolute()))
                            except:
                                st.info(f"{folder_name} payloads folder: {folder_path.absolute()}")

                # Enhanced Text Info Section (requires Docling payload)
                if payload_exists:
                    st.markdown("---")
                    st.subheader("📊 Enhanced Text Info (Color & Edge Analysis)")

                    # Check if enhanced text info exists
                    enhanced_path = ENHANCED_TEXT_INFO_PATH / f"{pdf_path.stem}_enhanced_text_info.json"
                    enhanced_exists = enhanced_path.exists()

                    enhanced_col1, enhanced_col2 = st.columns([2, 1])

                    with enhanced_col1:
                        if enhanced_exists:
                            st.success(f"✅ Enhanced text info exists: `{enhanced_path.name}`")

                            # Load and display metadata
                            try:
                                with open(enhanced_path, 'r', encoding='utf-8') as f:
                                    enhanced_data = json.load(f)

                                # Display summary statistics
                                st.markdown("**Analysis Summary:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Cells Analyzed", enhanced_data.get("total_cells", 0))
                                with col2:
                                    st.metric("Elements Processed", len(enhanced_data.get("elements", [])))
                                with col3:
                                    match_rate = enhanced_data.get("match_rate", "0%")
                                    st.metric("Match Rate", match_rate)

                                st.caption(f"File size: {enhanced_path.stat().st_size / 1024:.1f} KB")

                                # Show what analysis includes
                                st.markdown("**Includes:**")
                                st.write("- K-means color clustering (foreground/background)")
                                st.write("- Sobel gradient analysis (full + foreground-only)")
                                st.write("- Laplacian edge detection")
                                st.write("- Local variance (texture analysis)")
                                st.write("- Bilateral filter response")
                                st.write("- ELA (Error Level Analysis)")

                            except Exception as e:
                                st.warning(f"⚠️ Could not load enhanced text info: {str(e)}")
                        else:
                            st.info("ℹ️ No enhanced text info exists for this document yet")
                            st.caption("Enhanced text info includes:")
                            st.caption("- Color clustering analysis (foreground/background separation)")
                            st.caption("- Edge/gradient statistics (Sobel, Laplacian, ELA)")
                            st.caption("- Cell-to-span matching with PyMuPDF")
                            st.caption(f"Will be saved to: `{enhanced_path.name}`")

                    with enhanced_col2:
                        # Generate Enhanced Text Info button
                        if enhanced_exists:
                            enhanced_button_label = "🔄 Regenerate Enhanced Text Info"
                            enhanced_force_regen = True
                        else:
                            enhanced_button_label = "📊 Generate Enhanced Text Info"
                            enhanced_force_regen = False

                        if st.button(enhanced_button_label, use_container_width=True, key="generate_enhanced"):
                            with st.spinner("Generating enhanced text info... This may take several minutes (color clustering + edge analysis)."):
                                success, message, result_path = generate_enhanced_text_info(
                                    pdf_path,
                                    force_regenerate=enhanced_force_regen
                                )

                                if success:
                                    st.success(message)
                                    st.rerun()  # Refresh to show new data
                                else:
                                    st.error(message)

                        # Open Enhanced Text Info folder button
                        if st.button("📁 Open Enhanced Text Info Folder", use_container_width=True, key="open_enhanced_folder"):
                            # Create folder if it doesn't exist
                            ENHANCED_TEXT_INFO_PATH.mkdir(parents=True, exist_ok=True)
                            # Open in file explorer (Windows)
                            try:
                                os.startfile(str(ENHANCED_TEXT_INFO_PATH.absolute()))
                            except:
                                st.info(f"Enhanced text info folder: {ENHANCED_TEXT_INFO_PATH.absolute()}")

                    # Cluster Analysis Section (requires enhanced text info)
                    if enhanced_exists and HAS_CLUSTER_ANALYSIS:
                        st.markdown("#### 📈 Cluster Analysis")

                        # Check if cluster results exist
                        cluster_result_path = CLUSTER_RESULTS_PATH / f"{enhanced_path.stem}_cluster_result.pkl"
                        cluster_exists = cluster_result_path.exists()

                        if not cluster_exists:
                            st.info("ℹ️ No cluster analysis exists yet")
                            st.caption("Cluster analysis includes:")
                            st.caption("- RGB to single number conversion (luminance + hex decimal)")
                            st.caption("- Yeo-Johnson transformation for normality")
                            st.caption("- Feature standardization")
                            st.caption("- K-means clustering with optimal K selection (Silhouette Score)")
                            st.caption("- DBSCAN with automatic eps determination")

                            if st.button("📈 Perform Cluster Analysis", use_container_width=True, key="cluster_analysis_docling"):
                                with st.spinner("Performing cluster analysis... This may take a minute."):
                                    success, message, result = perform_and_save_cluster_analysis(
                                        enhanced_text_info_path=enhanced_path,
                                        use_gpu=True,
                                        max_k=4
                                    )

                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)
                        else:
                            # Load cluster results
                            import pickle
                            with open(cluster_result_path, 'rb') as f:
                                cluster_result = pickle.load(f)

                            st.success(f"✅ Cluster analysis complete")

                            # Display Hopkins statistic (if available)
                            has_hopkins = hasattr(cluster_result, 'hopkins_statistic')
                            if has_hopkins:
                                hopkins_val = cluster_result.hopkins_statistic
                                n_samples = len(cluster_result.texts)

                                # Dynamic thresholds based on sample size
                                if n_samples < 30:
                                    threshold = 0.75
                                    size_context = "small dataset (n<30)"
                                elif n_samples < 100:
                                    threshold = 0.70
                                    size_context = "medium dataset (30-100)"
                                else:
                                    threshold = 0.65
                                    size_context = "large dataset (100+)"

                                if hopkins_val < 0.3:
                                    st.warning(f"⚠️ **Hopkins Statistic: {hopkins_val:.3f}** ({size_context}) - Data appears uniformly distributed. Clustering may not be appropriate.")
                                elif hopkins_val < threshold:
                                    st.warning(f"⚠️ **Hopkins Statistic: {hopkins_val:.3f}** ({size_context}) - Threshold is {threshold:.2f} for this sample size. Clustering results may not be meaningful.")
                                else:
                                    st.info(f"✓ **Hopkins Statistic: {hopkins_val:.3f}** ({size_context}) - Data shows clustering tendency (>{threshold:.2f}).")
                            else:
                                st.warning("⚠️ Old cluster result detected. Regenerate to see Hopkins statistic (clustering tendency test).")
                                if st.button("🔄 Regenerate Now", use_container_width=True, key="regen_cluster_prompt_docling"):
                                    with st.spinner("Regenerating cluster analysis..."):
                                        success, message, result = perform_and_save_cluster_analysis(
                                            enhanced_text_info_path=enhanced_path,
                                            use_gpu=True,
                                            max_k=4
                                        )

                                        if success:
                                            st.success(message)
                                            st.rerun()
                                        else:
                                            st.error(message)

                            # Display K-means and DBSCAN statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**K-means:**")
                                st.write(f"- Clusters: {cluster_result.kmeans_n_clusters}")
                                st.write(f"- Silhouette: {cluster_result.kmeans_silhouette:.3f}")
                            with col2:
                                st.markdown("**DBSCAN:**")
                                st.write(f"- Clusters: {cluster_result.dbscan_n_clusters}")
                                if cluster_result.dbscan_silhouette:
                                    st.write(f"- Silhouette: {cluster_result.dbscan_silhouette:.3f}")
                                st.write(f"- Eps: {cluster_result.dbscan_eps:.3f}")

                            # Anomaly Detection Results
                            has_anomaly_detection = hasattr(cluster_result, 'ensemble_anomaly_flags')
                            if SHOW_ANOMALY_DETECTION and has_anomaly_detection:
                                st.markdown("---")
                                st.markdown("### 🔍 Anomaly Detection")
                                st.caption(f"⚙️ Feature extraction DPI: {FEATURE_EXTRACTION_DPI} | Contamination: 3% | Feature threshold: >2.5σ")

                                n_anomalies = int(np.sum(cluster_result.ensemble_anomaly_flags))
                                anomaly_rate = float(np.mean(cluster_result.ensemble_anomaly_flags)) * 100
                                high_confidence = int(np.sum(cluster_result.ensemble_agreement_count == 3))
                                medium_confidence = int(np.sum(cluster_result.ensemble_agreement_count == 2))

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Anomalies", n_anomalies, f"{anomaly_rate:.1f}%")
                                with col2:
                                    st.metric("High Confidence", high_confidence, "3/3 methods")
                                with col3:
                                    st.metric("Medium Confidence", medium_confidence, "2/3 methods")

                                if n_anomalies > 0:
                                    with st.expander(f"🚨 Anomalous Text Elements ({n_anomalies} items)", expanded=True):
                                        # Get anomaly indices
                                        anomaly_indices = np.where(cluster_result.ensemble_anomaly_flags)[0]

                                        # Sort by confidence (agreement count descending)
                                        sorted_indices = sorted(
                                            anomaly_indices,
                                            key=lambda idx: cluster_result.ensemble_agreement_count[idx],
                                            reverse=True
                                        )

                                        # Display each anomaly
                                        for rank, idx in enumerate(sorted_indices[:20], 1):  # Limit to top 20
                                            agreement = cluster_result.ensemble_agreement_count[idx]
                                            text = cluster_result.texts[idx]

                                            # Confidence level
                                            if agreement == 3:
                                                confidence_badge = "🔴 **HIGH** (3/3 methods)"
                                                confidence_color = "#ff4b4b"
                                            else:
                                                confidence_badge = "🟡 **MEDIUM** (2/3 methods)"
                                                confidence_color = "#ffa500"

                                            # Display anomaly
                                            st.markdown(f"**#{rank}** {confidence_badge}")
                                            st.markdown(f"> *\"{text}\"*")

                                            # Get feature values for this text
                                            features = cluster_result.features_standardized[idx]

                                            # Show which features are most anomalous (>2.5 std from mean for high confidence)
                                            anomalous_features = []
                                            for feat_idx, feat_name in enumerate(cluster_result.feature_names):
                                                feat_value = features[feat_idx]
                                                if abs(feat_value) > 2.5:  # More than 2.5 std from mean (reduces false positives)
                                                    anomalous_features.append((feat_name, feat_value))

                                            if anomalous_features:
                                                st.markdown("**Unusual characteristics:**")
                                                for feat_name, feat_value in sorted(anomalous_features, key=lambda x: abs(x[1]), reverse=True)[:3]:
                                                    direction = "much higher" if feat_value > 0 else "much lower"

                                                    # Human-readable explanation
                                                    explanation = get_anomaly_explanation(feat_name, feat_value)
                                                    st.markdown(f"- **{feat_name}**: {explanation} ({feat_value:+.2f} σ from typical)")
                                            else:
                                                st.markdown("*No specific features stand out as unusual*")

                                            st.markdown("---")

                                        if n_anomalies > 20:
                                            st.caption(f"... and {n_anomalies - 20} more anomalies")
                                else:
                                    st.success("✓ No significant anomalies detected")
                            elif SHOW_ANOMALY_DETECTION:
                                st.warning("⚠️ Old cluster result detected. Regenerate to see anomaly detection.")
                                if st.button("🔄 Regenerate for Anomaly Detection", use_container_width=True, key="regen_anomaly_docling"):
                                    with st.spinner("Regenerating cluster analysis..."):
                                        success, message, result = perform_and_save_cluster_analysis(
                                            enhanced_text_info_path=enhanced_path,
                                            use_gpu=True,
                                            max_k=4
                                        )

                                        if success:
                                            st.success(message)
                                            st.rerun()
                                        else:
                                            st.error(message)

                            # Cluster selection
                            st.markdown("**Select Cluster:**")
                            cluster_method = st.selectbox(
                                "Clustering Method",
                                ["K-means", "DBSCAN"],
                                key="cluster_method_docling"
                            )

                            if cluster_method == "K-means":
                                cluster_labels = cluster_result.kmeans_labels
                                unique_clusters = sorted(set(cluster_labels))
                                selected_cluster = st.selectbox(
                                    "Cluster",
                                    unique_clusters,
                                    format_func=lambda x: f"Cluster {x}",
                                    key="selected_cluster_kmeans_docling"
                                )
                            else:  # DBSCAN
                                cluster_labels = cluster_result.dbscan_labels
                                unique_clusters = sorted(set(cluster_labels))
                                selected_cluster = st.selectbox(
                                    "Cluster",
                                    unique_clusters,
                                    format_func=lambda x: "Noise" if x == -1 else f"Cluster {x}",
                                    key="selected_cluster_dbscan_docling"
                                )

                            # Get texts in selected cluster
                            cluster_mask = cluster_labels == selected_cluster
                            cluster_texts = [text for i, text in enumerate(cluster_result.texts) if cluster_mask[i]]
                            cluster_features = cluster_result.features_standardized[cluster_mask]

                            # Display cluster information
                            st.markdown(f"**Cluster {selected_cluster if selected_cluster != -1 else 'Noise'} Details:**")

                            # Number of items
                            st.metric("Items in Cluster", len(cluster_texts))

                            # Show texts (limit to first 50 for display)
                            with st.expander(f"📝 Texts in Cluster ({len(cluster_texts)} items)", expanded=True):
                                for i, text in enumerate(cluster_texts[:50]):
                                    st.write(f"{i+1}. {text}")
                                if len(cluster_texts) > 50:
                                    st.caption(f"... and {len(cluster_texts) - 50} more items")

                            # Cluster statistics
                            if len(cluster_features) > 0:
                                with st.expander("📊 Cluster Statistics", expanded=False):
                                    st.markdown("**Feature Statistics (Standardized Space):**")

                                    # Calculate mean and std for this cluster
                                    cluster_mean = np.mean(cluster_features, axis=0)
                                    cluster_std = np.std(cluster_features, axis=0)
                                    global_mean = np.mean(cluster_result.features_standardized, axis=0)

                                    # Create comparison table
                                    stats_data = []
                                    for i, feature_name in enumerate(cluster_result.feature_names):
                                        stats_data.append({
                                            "Feature": feature_name,
                                            "Cluster Mean": f"{cluster_mean[i]:.3f}",
                                            "Global Mean": f"{global_mean[i]:.3f}",
                                            "Cluster Std": f"{cluster_std[i]:.3f}"
                                        })

                                    st.dataframe(stats_data, use_container_width=True)

                            # Feature importance for separation
                            if selected_cluster != -1:  # Only for non-noise clusters
                                feature_importance = cluster_result.get_feature_importance_for_separation(
                                    labels=cluster_labels,
                                    cluster_id=selected_cluster,
                                    top_n=3
                                )

                                if feature_importance:
                                    nearest_cluster = cluster_result.find_nearest_neighbor_cluster(cluster_labels, selected_cluster)
                                    st.markdown(f"**🎯 What Makes This Cluster Different from Cluster {nearest_cluster}:**")
                                    st.caption("Look for these differences when examining the text:")

                                    # Get centroids for explanation
                                    cluster_centroid = cluster_result.get_cluster_centroids(cluster_labels, selected_cluster)
                                    neighbor_centroid = cluster_result.get_cluster_centroids(cluster_labels, nearest_cluster)

                                    for i, (feature_name, abs_diff, percentage) in enumerate(feature_importance, 1):
                                        # Find feature index
                                        feature_idx = cluster_result.feature_names.index(feature_name)

                                        # Get human-readable explanation
                                        explanation = get_human_explanation_for_feature(
                                            feature_name,
                                            cluster_centroid[feature_idx],
                                            neighbor_centroid[feature_idx]
                                        )

                                        st.markdown(f"**{i}. {explanation}**")
                                        st.caption(f"   Contributes {percentage:.1f}% to cluster separation")

                            # Regenerate button
                            if st.button("🔄 Regenerate Cluster Analysis", use_container_width=True, key="regen_cluster_docling"):
                                with st.spinner("Regenerating cluster analysis..."):
                                    success, message, result = perform_and_save_cluster_analysis(
                                        enhanced_text_info_path=enhanced_path,
                                        use_gpu=True,
                                        max_k=4
                                    )

                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)

        # Azure DI Section (for non-OCRed documents)
        if not has_text and is_azure_di_api_enabled():
            st.markdown("---")
            st.subheader("☁️ Azure Document Intelligence (Non-OCRed)")

            # Check if Azure DI payload exists
            azure_di_path = AZURE_DI_PAYLOADS_PATH / f"{pdf_path.stem}_Azure_DI_payload.json"
            azure_di_exists = azure_di_path.exists()

            azure_di_col1, azure_di_col2 = st.columns([2, 1])

            with azure_di_col1:
                if azure_di_exists:
                    st.success(f"✅ Azure DI payload exists: `{azure_di_path.name}`")

                    # Load and display metadata
                    try:
                        with open(azure_di_path, 'r', encoding='utf-8') as f:
                            azure_di_data = json.load(f)

                        metadata = azure_di_data.get('metadata', {})
                        st.markdown("**Processing Info:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Pages", metadata.get('page_count', 0))
                        with col2:
                            st.metric("Processing Time", f"{metadata.get('processing_time', 0):.1f}s")

                        st.caption(f"File size: {azure_di_path.stat().st_size / 1024:.1f} KB")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load Azure DI payload: {str(e)}")
                else:
                    st.info("ℹ️ No Azure DI payload exists yet")
                    st.caption("Azure DI will perform OCR on this scanned document")
                    st.caption(f"Will be saved to: `{azure_di_path.name}`")
                    st.caption(f"Folder: `{azure_di_path.parent.absolute()}`")

            with azure_di_col2:
                # Generate Azure DI button
                if azure_di_exists:
                    azure_di_button_label = "🔄 Regenerate Azure DI Payload"
                    azure_di_force_regen = True
                else:
                    azure_di_button_label = "☁️ Generate Azure DI Payload"
                    azure_di_force_regen = False

                if st.button(azure_di_button_label, use_container_width=True, key="generate_azure_di"):
                    with st.spinner("Generating Azure DI payload... This may take a minute."):
                        success, message, result_path = generate_azure_di_payload(
                            pdf_path,
                            force_regenerate=azure_di_force_regen
                        )

                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

                # Open Azure DI folder button
                if st.button("📁 Open Azure DI Payloads Folder", use_container_width=True, key="open_azure_di_folder"):
                    AZURE_DI_PAYLOADS_PATH.mkdir(parents=True, exist_ok=True)
                    try:
                        os.startfile(str(AZURE_DI_PAYLOADS_PATH.absolute()))
                    except:
                        st.info(f"Azure DI payloads folder: {AZURE_DI_PAYLOADS_PATH.absolute()}")

            # Enhanced Text Info for Azure DI
            if azure_di_exists:
                st.markdown("#### 📊 Enhanced Text Info (Color & Edge Analysis)")

                enhanced_azure_di_path = ENHANCED_TEXT_INFO_PATH / f"{pdf_path.stem}_Azure_DI_enhanced_text_info.json"
                enhanced_azure_di_exists = enhanced_azure_di_path.exists()

                enh_col1, enh_col2 = st.columns([2, 1])

                with enh_col1:
                    if enhanced_azure_di_exists:
                        st.success(f"✅ Enhanced text info exists: `{enhanced_azure_di_path.name}`")

                        try:
                            with open(enhanced_azure_di_path, 'r', encoding='utf-8') as f:
                                enhanced_data = json.load(f)

                            st.markdown("**Analysis Summary:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Cells Analyzed", enhanced_data.get("total_cells", 0))
                            with col2:
                                st.metric("Elements Processed", len(enhanced_data.get("elements", [])))

                            st.caption(f"File size: {enhanced_azure_di_path.stat().st_size / 1024:.1f} KB")
                        except Exception as e:
                            st.warning(f"⚠️ Could not load enhanced text info: {str(e)}")
                    else:
                        st.info("ℹ️ No enhanced text info exists yet")
                        st.caption(f"Will be saved to: `{enhanced_azure_di_path.name}`")

                with enh_col2:
                    if enhanced_azure_di_exists:
                        enhanced_azure_di_button_label = "🔄 Regenerate Enhanced Text Info"
                        enhanced_azure_di_force_regen = True
                    else:
                        enhanced_azure_di_button_label = "📊 Generate Enhanced Text Info"
                        enhanced_azure_di_force_regen = False

                    if st.button(enhanced_azure_di_button_label, use_container_width=True, key="generate_azure_di_enhanced"):
                        with st.spinner("Generating enhanced text info... This may take several minutes."):
                            success, message, result_path = generate_enhanced_text_info_from_azure_di(
                                pdf_path,
                                force_regenerate=enhanced_azure_di_force_regen
                            )

                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

                # Cluster Analysis for Azure DI (requires enhanced text info)
                if enhanced_azure_di_exists and HAS_CLUSTER_ANALYSIS:
                    st.markdown("#### 📈 Cluster Analysis")

                    # Check if cluster results exist
                    cluster_result_path = CLUSTER_RESULTS_PATH / f"{enhanced_azure_di_path.stem}_cluster_result.pkl"
                    cluster_exists = cluster_result_path.exists()

                    if not cluster_exists:
                        st.info("ℹ️ No cluster analysis exists yet")
                        st.caption("Cluster analysis includes:")
                        st.caption("- RGB to single number conversion (luminance + hex decimal)")
                        st.caption("- Yeo-Johnson transformation for normality")
                        st.caption("- Feature standardization")
                        st.caption("- K-means clustering with optimal K selection (Silhouette Score)")
                        st.caption("- DBSCAN with automatic eps determination")

                        if st.button("📈 Perform Cluster Analysis", use_container_width=True, key="cluster_analysis_azure_di"):
                            with st.spinner("Performing cluster analysis... This may take a minute."):
                                success, message, result = perform_and_save_cluster_analysis(
                                    enhanced_text_info_path=enhanced_azure_di_path,
                                    use_gpu=True,
                                    max_k=4
                                )

                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    else:
                        # Load cluster results
                        import pickle
                        with open(cluster_result_path, 'rb') as f:
                            cluster_result = pickle.load(f)

                        st.success(f"✅ Cluster analysis complete")

                        # Display Hopkins statistic (if available)
                        has_hopkins = hasattr(cluster_result, 'hopkins_statistic')
                        if has_hopkins:
                            hopkins_val = cluster_result.hopkins_statistic
                            n_samples = len(cluster_result.texts)

                            # Dynamic thresholds based on sample size
                            if n_samples < 30:
                                threshold = 0.75
                                size_context = "small dataset (n<30)"
                            elif n_samples < 100:
                                threshold = 0.70
                                size_context = "medium dataset (30-100)"
                            else:
                                threshold = 0.65
                                size_context = "large dataset (100+)"

                            if hopkins_val < 0.3:
                                st.warning(f"⚠️ **Hopkins Statistic: {hopkins_val:.3f}** ({size_context}) - Data appears uniformly distributed. Clustering may not be appropriate.")
                            elif hopkins_val < threshold:
                                st.warning(f"⚠️ **Hopkins Statistic: {hopkins_val:.3f}** ({size_context}) - Threshold is {threshold:.2f} for this sample size. Clustering results may not be meaningful.")
                            else:
                                st.info(f"✓ **Hopkins Statistic: {hopkins_val:.3f}** ({size_context}) - Data shows clustering tendency (>{threshold:.2f}).")
                        else:
                            st.warning("⚠️ Old cluster result detected. Regenerate to see Hopkins statistic (clustering tendency test).")
                            if st.button("🔄 Regenerate Now", use_container_width=True, key="regen_cluster_prompt_azure_di"):
                                with st.spinner("Regenerating cluster analysis..."):
                                    success, message, result = perform_and_save_cluster_analysis(
                                        enhanced_text_info_path=enhanced_azure_di_path,
                                        use_gpu=True,
                                        max_k=4
                                    )

                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)

                        # Display K-means and DBSCAN statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**K-means:**")
                            st.write(f"- Clusters: {cluster_result.kmeans_n_clusters}")
                            st.write(f"- Silhouette: {cluster_result.kmeans_silhouette:.3f}")
                        with col2:
                            st.markdown("**DBSCAN:**")
                            st.write(f"- Clusters: {cluster_result.dbscan_n_clusters}")
                            if cluster_result.dbscan_silhouette:
                                st.write(f"- Silhouette: {cluster_result.dbscan_silhouette:.3f}")
                            st.write(f"- Eps: {cluster_result.dbscan_eps:.3f}")

                        # Anomaly Detection Results
                        has_anomaly_detection = hasattr(cluster_result, 'ensemble_anomaly_flags')
                        if SHOW_ANOMALY_DETECTION and has_anomaly_detection:
                            st.markdown("---")
                            st.markdown("### 🔍 Anomaly Detection")

                            # Display DPI information (check if extraction_dpi is in match_statistics)
                            extraction_dpi_used = cluster_result.match_statistics.get('extraction_dpi', FEATURE_EXTRACTION_DPI)
                            effective_dpi_calculated = cluster_result.match_statistics.get('effective_dpi')

                            if effective_dpi_calculated:
                                dpi_caption = f"⚙️ Extraction DPI: {extraction_dpi_used} (effective: {int(effective_dpi_calculated)}, max: {FEATURE_EXTRACTION_DPI})"
                            else:
                                dpi_caption = f"⚙️ Feature extraction DPI: {extraction_dpi_used}"

                            st.caption(f"{dpi_caption} | Contamination: 3% | Feature threshold: >2.5σ")

                            n_anomalies = int(np.sum(cluster_result.ensemble_anomaly_flags))
                            anomaly_rate = float(np.mean(cluster_result.ensemble_anomaly_flags)) * 100
                            high_confidence = int(np.sum(cluster_result.ensemble_agreement_count == 3))
                            medium_confidence = int(np.sum(cluster_result.ensemble_agreement_count == 2))

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Anomalies", n_anomalies, f"{anomaly_rate:.1f}%")
                            with col2:
                                st.metric("High Confidence", high_confidence, "3/3 methods")
                            with col3:
                                st.metric("Medium Confidence", medium_confidence, "2/3 methods")

                            if n_anomalies > 0:
                                with st.expander(f"🚨 Anomalous Text Elements ({n_anomalies} items)", expanded=True):
                                    # Get anomaly indices
                                    anomaly_indices = np.where(cluster_result.ensemble_anomaly_flags)[0]

                                    # Sort by confidence (agreement count descending)
                                    sorted_indices = sorted(
                                        anomaly_indices,
                                        key=lambda idx: cluster_result.ensemble_agreement_count[idx],
                                        reverse=True
                                    )

                                    # Display each anomaly
                                    for rank, idx in enumerate(sorted_indices[:20], 1):  # Limit to top 20
                                        agreement = cluster_result.ensemble_agreement_count[idx]
                                        text = cluster_result.texts[idx]

                                        # Confidence level
                                        if agreement == 3:
                                            confidence_badge = "🔴 **HIGH** (3/3 methods)"
                                            confidence_color = "#ff4b4b"
                                        else:
                                            confidence_badge = "🟡 **MEDIUM** (2/3 methods)"
                                            confidence_color = "#ffa500"

                                        # Display anomaly
                                        st.markdown(f"**#{rank}** {confidence_badge}")
                                        st.markdown(f"> *\"{text}\"*")

                                        # Get feature values for this text
                                        features = cluster_result.features_standardized[idx]

                                        # Show which features are most anomalous (>2.5 std from mean for high confidence)
                                        anomalous_features = []
                                        for feat_idx, feat_name in enumerate(cluster_result.feature_names):
                                            feat_value = features[feat_idx]
                                            if abs(feat_value) > 2.5:  # More than 2.5 std from mean (reduces false positives)
                                                anomalous_features.append((feat_name, feat_value))

                                        if anomalous_features:
                                            st.markdown("**Unusual characteristics:**")
                                            for feat_name, feat_value in sorted(anomalous_features, key=lambda x: abs(x[1]), reverse=True)[:3]:
                                                direction = "much higher" if feat_value > 0 else "much lower"

                                                # Human-readable explanation
                                                explanation = get_anomaly_explanation(feat_name, feat_value)
                                                st.markdown(f"- **{feat_name}**: {explanation} ({feat_value:+.2f} σ from typical)")
                                        else:
                                            st.markdown("*No specific features stand out as unusual*")

                                        st.markdown("---")

                                    if n_anomalies > 20:
                                        st.caption(f"... and {n_anomalies - 20} more anomalies")
                            else:
                                st.success("✓ No significant anomalies detected")
                        elif SHOW_ANOMALY_DETECTION:
                            st.warning("⚠️ Old cluster result detected. Regenerate to see anomaly detection.")
                            if st.button("🔄 Regenerate for Anomaly Detection", use_container_width=True, key="regen_anomaly_azure_di"):
                                with st.spinner("Regenerating cluster analysis..."):
                                    success, message, result = perform_and_save_cluster_analysis(
                                        enhanced_text_info_path=enhanced_azure_di_path,
                                        use_gpu=True,
                                        max_k=4
                                    )

                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)

                        # Cluster selection
                        st.markdown("**Select Cluster:**")
                        cluster_method = st.selectbox(
                            "Clustering Method",
                            ["K-means", "DBSCAN"],
                            key="cluster_method_azure_di"
                        )

                        if cluster_method == "K-means":
                            cluster_labels = cluster_result.kmeans_labels
                            unique_clusters = sorted(set(cluster_labels))
                            selected_cluster = st.selectbox(
                                "Cluster",
                                unique_clusters,
                                format_func=lambda x: f"Cluster {x}",
                                key="selected_cluster_kmeans_azure_di"
                            )
                        else:  # DBSCAN
                            cluster_labels = cluster_result.dbscan_labels
                            unique_clusters = sorted(set(cluster_labels))
                            selected_cluster = st.selectbox(
                                "Cluster",
                                unique_clusters,
                                format_func=lambda x: "Noise" if x == -1 else f"Cluster {x}",
                                key="selected_cluster_dbscan_azure_di"
                            )

                        # Get texts in selected cluster
                        cluster_mask = cluster_labels == selected_cluster
                        cluster_texts = [text for i, text in enumerate(cluster_result.texts) if cluster_mask[i]]
                        cluster_features = cluster_result.features_standardized[cluster_mask]

                        # Display cluster information
                        st.markdown(f"**Cluster {selected_cluster if selected_cluster != -1 else 'Noise'} Details:**")

                        # Number of items
                        st.metric("Items in Cluster", len(cluster_texts))

                        # Show texts (limit to first 50 for display)
                        with st.expander(f"📝 Texts in Cluster ({len(cluster_texts)} items)", expanded=True):
                            for i, text in enumerate(cluster_texts[:50]):
                                st.write(f"{i+1}. {text}")
                            if len(cluster_texts) > 50:
                                st.caption(f"... and {len(cluster_texts) - 50} more items")

                        # Cluster statistics
                        if len(cluster_features) > 0:
                            with st.expander("📊 Cluster Statistics", expanded=False):
                                st.markdown("**Feature Statistics (Standardized Space):**")

                                # Calculate mean and std for this cluster
                                cluster_mean = np.mean(cluster_features, axis=0)
                                cluster_std = np.std(cluster_features, axis=0)
                                global_mean = np.mean(cluster_result.features_standardized, axis=0)

                                # Create comparison table
                                stats_data = []
                                for i, feature_name in enumerate(cluster_result.feature_names):
                                    stats_data.append({
                                        "Feature": feature_name,
                                        "Cluster Mean": f"{cluster_mean[i]:.3f}",
                                        "Global Mean": f"{global_mean[i]:.3f}",
                                        "Cluster Std": f"{cluster_std[i]:.3f}"
                                    })

                                st.dataframe(stats_data, use_container_width=True)

                        # Feature importance for separation
                        if selected_cluster != -1:  # Only for non-noise clusters
                            feature_importance = cluster_result.get_feature_importance_for_separation(
                                labels=cluster_labels,
                                cluster_id=selected_cluster,
                                top_n=3
                            )

                            if feature_importance:
                                nearest_cluster = cluster_result.find_nearest_neighbor_cluster(cluster_labels, selected_cluster)
                                st.markdown(f"**🎯 What Makes This Cluster Different from Cluster {nearest_cluster}:**")
                                st.caption("Look for these differences when examining the text:")

                                # Get centroids for explanation
                                cluster_centroid = cluster_result.get_cluster_centroids(cluster_labels, selected_cluster)
                                neighbor_centroid = cluster_result.get_cluster_centroids(cluster_labels, nearest_cluster)

                                for i, (feature_name, abs_diff, percentage) in enumerate(feature_importance, 1):
                                    # Find feature index
                                    feature_idx = cluster_result.feature_names.index(feature_name)

                                    # Get human-readable explanation
                                    explanation = get_human_explanation_for_feature(
                                        feature_name,
                                        cluster_centroid[feature_idx],
                                        neighbor_centroid[feature_idx]
                                    )

                                    st.markdown(f"**{i}. {explanation}**")
                                    st.caption(f"   Contributes {percentage:.1f}% to cluster separation")

                        # Regenerate button
                        if st.button("🔄 Regenerate Cluster Analysis", use_container_width=True, key="regen_cluster_azure_di"):
                            with st.spinner("Regenerating cluster analysis..."):
                                success, message, result = perform_and_save_cluster_analysis(
                                    enhanced_text_info_path=enhanced_azure_di_path,
                                    use_gpu=True,
                                    max_k=4
                                )

                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)

        # Font Analysis Section (only if Docling payload exists and has word-level data, or Adobe payload exists, or pikepdf payload exists)
        if has_text:
            adobe_exists_for_analysis, adobe_path_for_analysis, _ = check_adobe_payload_exists(pdf_path)
            pikepdf_exists_for_analysis, pikepdf_path_for_analysis, _ = check_pikepdf_payload_exists(pdf_path)

            if (payload_exists and metadata and metadata.get("has_word_level")) or adobe_exists_for_analysis or pikepdf_exists_for_analysis:
                st.markdown("---")
                st.subheader("🔤 Font Analysis")

                # Analyze Docling payload if available
                font_analysis = None
                if payload_exists and metadata and metadata.get("has_word_level"):
                    with st.spinner("Analyzing Docling fonts..."):
                        font_analysis = analyze_fonts_in_payload(payload_path)

                # Analyze Adobe payload if available
                adobe_analysis = None
                if adobe_exists_for_analysis:
                    with st.spinner("Analyzing Adobe properties..."):
                        adobe_analysis = analyze_adobe_payload(adobe_path_for_analysis)

                # Analyze pikepdf payload if available (reuse adobe analyzer since structure is identical)
                pikepdf_analysis = None
                if pikepdf_exists_for_analysis:
                    with st.spinner("Analyzing pikepdf properties..."):
                        pikepdf_analysis = analyze_adobe_payload(pikepdf_path_for_analysis)

                # Create tabs conditionally based on available analyses
                tab1 = None
                tab2 = None
                tab3 = None

                # Determine which tabs to show
                tab_names = []
                if font_analysis:
                    tab_names.append("Docling Font Analysis")
                if adobe_analysis:
                    tab_names.append("Adobe Property Analysis")
                if pikepdf_analysis:
                    tab_names.append("pikepdf Property Analysis")

                if len(tab_names) == 3:
                    tab1, tab2, tab3 = st.tabs(tab_names)
                elif len(tab_names) == 2:
                    if font_analysis and adobe_analysis:
                        tab1, tab2 = st.tabs(tab_names)
                    elif font_analysis and pikepdf_analysis:
                        tab1, tab3 = st.tabs(tab_names)
                    else:  # adobe_analysis and pikepdf_analysis
                        tab2, tab3 = st.tabs(tab_names)
                elif len(tab_names) == 1:
                    if font_analysis:
                        tab1 = st.container()
                    elif adobe_analysis:
                        tab2 = st.container()
                    else:  # pikepdf_analysis
                        tab3 = st.container()
                else:
                    st.warning("⚠️ No analysis data available")

                # Docling Font Analysis Tab
                if tab1 is not None and font_analysis:
                    with tab1:
                        # Display overall statistics
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.metric("Total Unique Fonts", font_analysis["total_fonts"])
                        with stats_col2:
                            st.metric("Total Text Items", font_analysis["total_text_items"])

                        # Font statistics table
                        st.markdown("#### Font Usage Statistics")

                        font_stats = font_analysis["font_stats"]

                        # Create a formatted display of font statistics
                        if font_stats:
                            # Display as expandable table
                            with st.expander(f"📊 View All {len(font_stats)} Fonts", expanded=True):
                                # Create columns for better layout
                                for idx, (font_name, count) in enumerate(font_stats.items(), 1):
                                    percentage = (count / font_analysis["total_text_items"]) * 100
                                    st.write(f"**{idx}. {font_name}**: {count:,} items ({percentage:.1f}%)")

                        # Font filter selectbox (single selection)
                        st.markdown("#### Filter Text by Font")
                        st.write("Select a font to view all text items in that font:")

                        available_fonts = list(font_stats.keys())
                        selected_font = st.selectbox(
                            "Choose a font to display:",
                            options=["-- Select a font --"] + available_fonts,
                            index=0,
                            key="font_filter"
                        )

                        # Display text items for selected font
                        if selected_font and selected_font != "-- Select a font --":
                            st.markdown(f"#### Text Items in: {selected_font}")

                            font_items = font_analysis["font_items"]

                            for font_name in [selected_font]:
                                items = font_items.get(font_name, [])

                                if items:
                                    with st.expander(f"**{font_name}** ({len(items)} items)", expanded=True):
                                        # Group by page for better organization
                                        items_by_page = {}
                                        for item in items:
                                            page = item["page"]
                                            if page not in items_by_page:
                                                items_by_page[page] = []
                                            items_by_page[page].append(item)

                                        # Display items grouped by page
                                        for page in sorted(items_by_page.keys()):
                                            st.markdown(f"**Page {page}:**")
                                            page_items = items_by_page[page]

                                            # Create a scrollable text area with all items
                                            text_display = []
                                            for idx, item in enumerate(page_items, 1):
                                                text = item["text"]
                                                font_key = item.get("font_key", "N/A")
                                                confidence = item.get("confidence")
                                                from_ocr = item.get("from_ocr", False)

                                                # Format with metadata
                                                meta_info = f"[Font Key: {font_key}"
                                                if confidence is not None:
                                                    meta_info += f", Conf: {confidence:.2f}"
                                                if from_ocr:
                                                    meta_info += ", OCR"
                                                meta_info += "]"

                                                text_display.append(f"{idx}. {text} {meta_info}")

                                            # Display in text area
                                            st.text_area(
                                                f"Items on page {page}",
                                                value="\n".join(text_display),
                                                height=min(300, max(100, len(page_items) * 20)),
                                                key=f"docling_font_{font_name}_page_{page}"
                                            )
                                            st.caption(f"{len(page_items)} items on this page")
                        else:
                            st.info("👆 Select a font from the dropdown above to view its text items")

                # Adobe Property Analysis Tab
                if tab2 is not None and adobe_analysis:
                    with tab2:
                        # Create property selectbox
                        st.markdown("#### Select Property to Analyze")

                        available_properties = []
                        if adobe_analysis["embedded"]["total"] > 0:
                            available_properties.append("Embedded")
                        if adobe_analysis["encoding"]["total"] > 0:
                            available_properties.append("Encoding")
                        if adobe_analysis["family_name"]["total"] > 0:
                            available_properties.append("Family Name")
                        if adobe_analysis["name"]["total"] > 0:
                            available_properties.append("Name")
                        if adobe_analysis["monospaced"]["total"] > 0:
                            available_properties.append("Monospaced")
                        if adobe_analysis["subset"]["total"] > 0:
                            available_properties.append("Subset")
                        if adobe_analysis["has_clip"]["total"] > 0:
                            available_properties.append("HasClip")
                        if adobe_analysis["text_size"]["total"] > 0:
                            available_properties.append("TextSize")
                        if adobe_analysis["line_height"]["total"] > 0:
                            available_properties.append("LineHeight")

                        if not available_properties:
                            st.warning("No properties found in Adobe payload")
                        else:
                            selected_property = st.selectbox(
                                "Choose a property to analyze:",
                                options=["-- Select a property --"] + available_properties,
                                index=0,
                                key="adobe_property_filter"
                            )

                            if selected_property and selected_property != "-- Select a property --":
                                # Map display name to internal key
                                property_map = {
                                    "Embedded": "embedded",
                                    "Encoding": "encoding",
                                    "Family Name": "family_name",
                                    "Name": "name",
                                    "Monospaced": "monospaced",
                                    "Subset": "subset",
                                    "HasClip": "has_clip",
                                    "TextSize": "text_size",
                                    "LineHeight": "line_height"
                                }

                                prop_key = property_map[selected_property]
                                prop_data = adobe_analysis[prop_key]

                                # Display statistics
                                st.markdown(f"#### {selected_property} Statistics")
                                st.metric("Total Items", prop_data["total"])

                                # Show value distribution
                                with st.expander(f"📊 {selected_property} Values", expanded=True):
                                    for value, count in prop_data["stats"].items():
                                        percentage = (count / prop_data["total"]) * 100
                                        display_value = format_rounding_classification(value) if prop_key in ("text_size", "line_height") else value
                                        st.write(f"**{display_value}**: {count:,} items ({percentage:.1f}%)")

                                # Value selectbox
                                st.markdown(f"#### Filter by {selected_property} Value")
                                available_values = list(prop_data["stats"].keys())
                                selected_value = st.selectbox(
                                    f"Choose a {selected_property.lower()} value:",
                                    options=["-- Select a value --"] + available_values,
                                    index=0,
                                    key=f"adobe_{prop_key}_value",
                                    format_func=lambda v: format_rounding_classification(v) if v in ROUNDING_CLASSIFICATION_LABELS else v
                                )

                                # Display text items for selected value
                                if selected_value and selected_value != "-- Select a value --":
                                    items = prop_data["items"].get(selected_value, [])

                                    if items:
                                        display_selected_label = format_rounding_classification(selected_value) if selected_value in ROUNDING_CLASSIFICATION_LABELS else selected_value
                                        st.markdown(f"#### Text Items with {selected_property} = {display_selected_label}")

                                        with st.expander(f"{len(items)} items", expanded=True):
                                            # Group by page
                                            items_by_page = {}
                                            for item in items:
                                                page = item["page"]
                                                if page not in items_by_page:
                                                    items_by_page[page] = []
                                                items_by_page[page].append(item)

                                            # Display by page
                                            for page in sorted(items_by_page.keys()):
                                                st.markdown(f"**Page {page}:**")
                                                page_items = items_by_page[page]

                                                text_display = []
                                                for idx, item in enumerate(page_items, 1):
                                                    text = item["text"]
                                                    if prop_key == "text_size" and "exact_size" in item:
                                                        classification_label = item.get("classification_label") or format_rounding_classification(item.get("classification", ""))
                                                        text_display.append(f"{idx}. {text} [{classification_label}; Exact: {item['exact_size']}]")
                                                    elif prop_key == "line_height" and "exact_height" in item:
                                                        classification_label = item.get("classification_label") or format_rounding_classification(item.get("classification", ""))
                                                        text_display.append(f"{idx}. {text} [{classification_label}; Exact: {item['exact_height']}]")
                                                    else:
                                                        text_display.append(f"{idx}. {text}")

                                                st.text_area(
                                                    f"Items on page {page}",
                                                    value="\n".join(text_display),
                                                    height=min(300, max(100, len(page_items) * 20)),
                                                    key=f"adobe_{prop_key}_{selected_value}_page_{page}"
                                                )
                                                st.caption(f"{len(page_items)} items on this page")
                                    else:
                                        st.info("No items found for this value")
                                else:
                                    st.info("👆 Select a value from the dropdown above to view text items")
                            else:
                                st.info("👆 Select a property from the dropdown above to begin analysis")

                # pikepdf Property Analysis Tab
                if tab3 is not None and pikepdf_analysis:
                    with tab3:
                        # Create property selectbox
                        st.markdown("#### Select Property to Analyze")

                        available_properties = []
                        if pikepdf_analysis["embedded"]["total"] > 0:
                            available_properties.append("Embedded")
                        if pikepdf_analysis["encoding"]["total"] > 0:
                            available_properties.append("Encoding")
                        if pikepdf_analysis["family_name"]["total"] > 0:
                            available_properties.append("Family Name")
                        if pikepdf_analysis["name"]["total"] > 0:
                            available_properties.append("Name")
                        if pikepdf_analysis["monospaced"]["total"] > 0:
                            available_properties.append("Monospaced")
                        if pikepdf_analysis["subset"]["total"] > 0:
                            available_properties.append("Subset")
                        if pikepdf_analysis["has_clip"]["total"] > 0:
                            available_properties.append("HasClip")
                        if pikepdf_analysis["text_size"]["total"] > 0:
                            available_properties.append("TextSize")
                        if pikepdf_analysis["line_height"]["total"] > 0:
                            available_properties.append("LineHeight")

                        if not available_properties:
                            st.warning("No properties found in pikepdf payload")
                        else:
                            selected_property = st.selectbox(
                                "Choose a property to analyze:",
                                options=["-- Select a property --"] + available_properties,
                                index=0,
                                key="pikepdf_property_filter"
                            )

                            if selected_property and selected_property != "-- Select a property --":
                                # Map display name to internal key
                                property_map = {
                                    "Embedded": "embedded",
                                    "Encoding": "encoding",
                                    "Family Name": "family_name",
                                    "Name": "name",
                                    "Monospaced": "monospaced",
                                    "Subset": "subset",
                                    "HasClip": "has_clip",
                                    "TextSize": "text_size",
                                    "LineHeight": "line_height"
                                }

                                prop_key = property_map[selected_property]
                                prop_data = pikepdf_analysis[prop_key]

                                # Display statistics
                                st.markdown(f"#### {selected_property} Statistics")
                                st.metric("Total Items", prop_data["total"])

                                # Show value distribution
                                with st.expander(f"📊 {selected_property} Values", expanded=True):
                                    for value, count in prop_data["stats"].items():
                                        percentage = (count / prop_data["total"]) * 100
                                        display_value = format_rounding_classification(value) if prop_key in ("text_size", "line_height") else value
                                        st.write(f"**{display_value}**: {count:,} items ({percentage:.1f}%)")

                                # Value selectbox
                                st.markdown(f"#### Filter by {selected_property} Value")
                                available_values = list(prop_data["stats"].keys())
                                selected_value = st.selectbox(
                                    f"Choose a {selected_property.lower()} value:",
                                    options=["-- Select a value --"] + available_values,
                                    index=0,
                                    key=f"pikepdf_{prop_key}_value",
                                    format_func=lambda v: format_rounding_classification(v) if v in ROUNDING_CLASSIFICATION_LABELS else v
                                )

                                # Display text items for selected value
                                if selected_value and selected_value != "-- Select a value --":
                                    items = prop_data["items"].get(selected_value, [])

                                    if items:
                                        display_selected_label = format_rounding_classification(selected_value) if selected_value in ROUNDING_CLASSIFICATION_LABELS else selected_value
                                        st.markdown(f"#### Text Items with {selected_property} = {display_selected_label}")

                                        with st.expander(f"{len(items)} items", expanded=True):
                                            # Group by page
                                            items_by_page = {}
                                            for item in items:
                                                page = item["page"]
                                                if page not in items_by_page:
                                                    items_by_page[page] = []
                                                items_by_page[page].append(item)

                                            # Display by page
                                            for page in sorted(items_by_page.keys()):
                                                st.markdown(f"**Page {page}:**")
                                                page_items = items_by_page[page]

                                                text_display = []
                                                for idx, item in enumerate(page_items, 1):
                                                    text = item["text"]
                                                    if prop_key == "text_size" and "exact_size" in item:
                                                        classification_label = item.get("classification_label") or format_rounding_classification(item.get("classification", ""))
                                                        text_display.append(f"{idx}. {text} [{classification_label}; Exact: {item['exact_size']}]")
                                                    elif prop_key == "line_height" and "exact_height" in item:
                                                        classification_label = item.get("classification_label") or format_rounding_classification(item.get("classification", ""))
                                                        text_display.append(f"{idx}. {text} [{classification_label}; Exact: {item['exact_height']}]")
                                                    else:
                                                        text_display.append(f"{idx}. {text}")

                                                st.text_area(
                                                    f"Items on page {page}",
                                                    value="\n".join(text_display),
                                                    height=min(300, max(100, len(page_items) * 20)),
                                                    key=f"pikepdf_{prop_key}_{selected_value}_page_{page}"
                                                )
                                                st.caption(f"{len(page_items)} items on this page")
                                    else:
                                        st.info("No items found for this value")
                                else:
                                    st.info("👆 Select a value from the dropdown above to view text items")
                            else:
                                st.info("👆 Select a property from the dropdown above to begin analysis")

                # Hidden Text Detection Section (only if payload exists and has word-level data)
                if payload_exists and metadata and metadata.get("has_word_level"):
                    st.markdown("---")
                    st.subheader("🔍 Hidden Text Detection")

                    st.write("Detect if text is hidden below other visible text (potential forgery indicator)")

                    if not HAS_FLATTENING:
                        st.warning("⚠️ Hidden text detection requires `pdf2image` and `img2pdf`. Install with: `pip install pdf2image img2pdf`")
                    else:
                        # Check if detection result already exists
                        detection_exists, detection_path, detection_info = check_overlapping_bboxes_exists(pdf_path)

                        detect_col1, detect_col2 = st.columns([3, 1])

                        with detect_col1:
                            if detection_exists:
                                st.success(f"✅ Detection result exists: `{detection_path.name}`")

                                # Show metadata
                                meta_col1, meta_col2 = st.columns(2)
                                with meta_col1:
                                    st.caption(f"Size: {detection_info['size_kb']:.1f} KB")
                                with meta_col2:
                                    st.caption(f"Modified: {detection_info['modified']}")
                            else:
                                st.info("""
                                **How it works:**
                                1. Checks original payload for overlapping text bboxes
                                2. Always creates flattened PDF (PDF → Images @ 400 DPI → PDF)
                                3. Generates payload for the flattened version
                                4. Compares payloads to detect hidden text in TWO ways:
                                   - Text with overlapping bboxes (one hidden, one visible)
                                   - Text in original but not in flattened (not rendered visually)
                                """)

                        with detect_col2:
                            if detection_exists:
                                if st.button("🔄 Regenerate", use_container_width=True, key="regen_detection"):
                                    with st.spinner("Detecting hidden text... This may take a minute."):
                                        detection_result = detect_hidden_text(pdf_path, payload_path)

                                        if detection_result:
                                            if "error" in detection_result:
                                                st.error(f"Error: {detection_result['error']}")
                                            else:
                                                # Store results in session state
                                                st.session_state['hidden_text_result'] = detection_result
                                                st.session_state['last_pdf_path'] = str(pdf_path)
                                                st.success("Detection regenerated!")
                                                st.rerun()

                                latest_visible_payload = get_latest_visible_payload_path(pdf_path)
                                if latest_visible_payload and latest_visible_payload.exists():
                                    if st.button("↻ Reprocess", use_container_width=True, key="reprocess_detection"):
                                        with st.spinner("Reprocessing hidden text comparison..."):
                                            detection_result = detect_hidden_text(
                                                pdf_path,
                                                payload_path,
                                                reuse_visible_payload=True,
                                            )

                                            if detection_result:
                                                if "error" in detection_result:
                                                    st.error(f"Error: {detection_result['error']}")
                                                else:
                                                    st.session_state['hidden_text_result'] = detection_result
                                                    st.session_state['last_pdf_path'] = str(pdf_path)
                                                    st.success("Hidden text comparison reprocessed using existing flattened payload.")
                                                    st.rerun()
                            else:
                                if st.button("🔎 Detect Hidden Text", use_container_width=True):
                                    with st.spinner("Detecting hidden text... This may take a minute."):
                                        detection_result = detect_hidden_text(pdf_path, payload_path)

                                        if detection_result:
                                            if "error" in detection_result:
                                                st.error(f"Error: {detection_result['error']}")
                                            else:
                                                # Store results in session state
                                                st.session_state['hidden_text_result'] = detection_result
                                                st.session_state['last_pdf_path'] = str(pdf_path)
                                                st.rerun()

                        # Load existing result if available
                        if detection_exists and 'hidden_text_result' not in st.session_state:
                            try:
                                with open(detection_path, 'r', encoding='utf-8') as f:
                                    detection_result = json.load(f)
                                    st.session_state['hidden_text_result'] = detection_result
                                    st.session_state['last_pdf_path'] = str(pdf_path)
                            except Exception as e:
                                st.error(f"Error loading existing result: {e}")

                        # Display detection results if available
                        if 'hidden_text_result' in st.session_state and st.session_state.get('last_pdf_path') == str(pdf_path):
                            result = st.session_state['hidden_text_result']

                            st.markdown("---")
                            st.markdown("#### Detection Results")

                            segmentation_info = result.get("segmentation_analysis") or result.get("metadata", {}).get("flattening_segmentation_analysis")
                            if segmentation_info and segmentation_info.get("is_problem_case"):
                                st.warning("⚠️ Flattening introduced a significant segmentation shift. Results may require advanced matching or reprocessing.")

                            total_hidden = result.get("total_hidden", 0)

                            if total_hidden > 0:
                                st.error(f"⚠️ **{total_hidden} hidden text item(s) detected!**")

                                # Show diagnostic stats
                                debug_stats = result.get("debug_stats", {})
                                total_with_overlap = result.get("total_hidden_with_overlap", 0)
                                total_no_overlap = result.get("total_hidden_no_overlap", 0)

                                stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
                                with stat_col1:
                                    st.metric("Total Hidden", total_hidden)
                                with stat_col2:
                                    st.metric("With Overlap", total_with_overlap)
                                with stat_col3:
                                    st.metric("Without Overlap", total_no_overlap)
                                with stat_col4:
                                    st.metric("Total Overlaps", result.get("total_overlaps", 0))
                                with stat_col5:
                                    st.metric("Blank Cells Skipped", debug_stats.get('blank_cells_skipped', 0))

                                # Display hidden text items (separated by type)
                                st.markdown("#### Hidden Text Items")

                                # Get both types of hidden items
                                hidden_items_overlap = result.get("hidden_items_with_overlap", [])
                                hidden_items_no_overlap = result.get("hidden_items_no_overlap", [])

                                # For backwards compatibility, if the new fields don't exist, use the legacy field
                                if not hidden_items_overlap and not hidden_items_no_overlap:
                                    hidden_items_overlap = result.get("hidden_items", [])

                                # Display items with overlap (forgery via overlapping bboxes)
                                if hidden_items_overlap:
                                    st.markdown("##### 📍 Hidden Text via Overlapping Bounding Boxes")
                                    st.caption(f"{len(hidden_items_overlap)} item(s) detected where text is hidden beneath visible text")

                                    # Group by page
                                    items_by_page = {}
                                    for item in hidden_items_overlap:
                                        page = item.get("page", 0)
                                        if page not in items_by_page:
                                            items_by_page[page] = []
                                        items_by_page[page].append(item)

                                    # Display by page
                                    for page in sorted(items_by_page.keys()):
                                        page_items = items_by_page[page]

                                        with st.expander(f"**Page {page}** ({len(page_items)} hidden items)", expanded=True):
                                            for idx, item in enumerate(page_items, 1):
                                                st.markdown(f"**Item {idx}:**")

                                                item_col1, item_col2 = st.columns(2)

                                                with item_col1:
                                                    st.write("**Hidden Text:**")
                                                    st.code(item.get("hidden_text", ""), language=None)
                                                    st.caption(f"Font: {item.get('hidden_font', 'Unknown')}")

                                                with item_col2:
                                                    st.write("**Visible Text (on top):**")
                                                    st.code(item.get("visible_text", ""), language=None)
                                                    st.caption(f"Font: {item.get('visible_font', 'Unknown')}")

                                                st.write(f"**Overlap (IoU):** {item.get('iou', 0):.2%}")
                                                st.write(f"**Confidence:** {item.get('confidence', 'UNKNOWN')}")

                                                st.markdown("---")

                                # Display items without overlap (text not rendered visually)
                                # This detects hidden text like invisible rendering mode, white-on-white, etc.
                                if hidden_items_no_overlap:
                                    st.markdown("##### 🚫 Text Not Rendered Visually (Hidden)")
                                    st.caption(f"{len(hidden_items_no_overlap)} item(s) detected where text exists in PDF structure but is not visible")

                                    # Group by page
                                    items_by_page_no_overlap = {}
                                    for item in hidden_items_no_overlap:
                                        page = item.get("page", 0)
                                        if page not in items_by_page_no_overlap:
                                            items_by_page_no_overlap[page] = []
                                        items_by_page_no_overlap[page].append(item)

                                    # Display by page
                                    for page in sorted(items_by_page_no_overlap.keys()):
                                        page_items = items_by_page_no_overlap[page]

                                        with st.expander(f"**Page {page}** ({len(page_items)} non-visible items)", expanded=True):
                                            for idx, item in enumerate(page_items, 1):
                                                st.markdown(f"**Item {idx}:**")

                                                st.write("**Text in PDF Structure (not visible):**")
                                                st.code(item.get("text", ""), language=None)
                                                st.caption(f"Font: {item.get('font', 'Unknown')}")

                                                st.write(f"**Reason:** {item.get('reason', 'not_in_flattened')}")

                                                bbox = item.get("bbox", {})
                                                if bbox:
                                                    st.caption(f"BBox: ({bbox.get('x0', 0):.1f}, {bbox.get('y0', 0):.1f}, {bbox.get('x1', 0):.1f}, {bbox.get('y1', 0):.1f})")

                                                closest = item.get("closest_candidate")
                                                if closest:
                                                    candidate_text = closest.get("text", "")
                                                    if candidate_text:
                                                        st.caption(f"Closest candidate text: {candidate_text}")
                                                    similarity = closest.get("similarity")
                                                    lev_dist = closest.get("levenshtein_distance")
                                                    if similarity is not None or lev_dist is not None:
                                                        st.caption(
                                                            f"Levenshtein distance: {lev_dist} | Similarity: {similarity:.2f}"
                                                            if similarity is not None and lev_dist is not None
                                                            else (
                                                                f"Similarity: {similarity:.2f}"
                                                                if similarity is not None
                                                                else f"Levenshtein distance: {lev_dist}"
                                                            )
                                                        )
                                                    center_distance = closest.get("center_distance")
                                                    x_overlap = closest.get("x_overlap")
                                                    vertical_gap = closest.get("vertical_gap")
                                                    iou = closest.get("iou")
                                                    metrics = []
                                                    if center_distance is not None:
                                                        metrics.append(f"center Δ {center_distance:.2f}pt")
                                                    if x_overlap is not None:
                                                        metrics.append(f"x-overlap {x_overlap:.2f}pt")
                                                    if vertical_gap is not None:
                                                        metrics.append(f"vertical gap {vertical_gap:.2f}pt")
                                                    if iou is not None:
                                                        metrics.append(f"IoU {iou:.2f}")
                                                    if metrics:
                                                        st.caption(" | ".join(metrics))

                                                st.markdown("---")

                                # Show info about generated files
                                if result.get('visible_pdf') or result.get('visible_payload'):
                                    with st.expander("📁 Generated Files"):
                                        if result.get('visible_pdf'):
                                            visible_pdf_name = Path(result['visible_pdf']).name if isinstance(result['visible_pdf'], str) else result['visible_pdf'].name
                                            st.write(f"**Flattened PDF:** `{visible_pdf_name}`")
                                        if result.get('visible_payload'):
                                            visible_payload_name = Path(result['visible_payload']).name if isinstance(result['visible_payload'], str) else result['visible_payload'].name
                                            st.write(f"**Visible Payload:** `{visible_payload_name}`")
                                        st.caption("These files are saved in the Docling payloads folder")

                            else:
                                st.success("✅ No hidden text detected!")

                                # Show diagnostic stats
                                debug_stats = result.get("debug_stats", {})
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                with stat_col1:
                                    st.metric("Overlaps Checked", result.get('total_overlaps', 0))
                                with stat_col2:
                                    st.metric("Cell Pairs Analyzed", debug_stats.get('total_cell_pairs_checked', 0))
                                with stat_col3:
                                    st.metric("Blank Cells Skipped", debug_stats.get('blank_cells_skipped', 0))

                                # Show blank text locations if any
                                blank_locations = result.get("blank_text_locations", [])
                                if blank_locations:
                                    st.markdown("---")
                                    st.markdown("#### 📋 Blank Text Cells Detected")
                                    st.caption(f"Found {len(blank_locations)} blank text cells in the document (these were excluded from overlap detection)")
                                    zero_bbox_count = sum(1 for blank in blank_locations if blank.get("bbox_is_zero"))
                                    non_zero_bbox_count = len(blank_locations) - zero_bbox_count
                                    st.caption(f"Zero-bbox blank cells: {zero_bbox_count} | Non-zero blank cells: {non_zero_bbox_count}")

                                    # Group blank cells by page
                                    blank_by_page = {}
                                    for blank in blank_locations:
                                        page = blank["page"]
                                        if page not in blank_by_page:
                                            blank_by_page[page] = []
                                        blank_by_page[page].append(blank)

                                    with st.expander(f"📍 Blank Cell Locations ({len(blank_by_page)} pages)", expanded=False):
                                        for page in sorted(blank_by_page.keys()):
                                            page_blanks = blank_by_page[page]
                                            st.markdown(f"**Page {page}:** {len(page_blanks)} blank cells")

                                            for idx, blank in enumerate(page_blanks[:10], 1):  # Show max 10 per page
                                                bbox = blank["bbox"]
                                                st.caption(f"  {idx}. Cell {blank['cell_index']} - Font: {blank['font']} - BBox: ({bbox.get('x0', 0):.1f}, {bbox.get('y0', 0):.1f}, {bbox.get('x1', 0):.1f}, {bbox.get('y1', 0):.1f})")
                                                bbox_status = "zero" if blank.get("bbox_is_zero") else "non-zero"
                                                bbox_source = blank.get("bbox_source", "unknown")
                                                st.caption(f"    -> BBox status: {bbox_status} (source: {bbox_source})")

                                            if len(page_blanks) > 10:
                                                st.caption(f"  ... and {len(page_blanks) - 10} more blank cells on this page")

                                with st.expander("ℹ️ What does this mean?"):
                                    st.write("""
                                    - No text is hidden below other visible text
                                    - The document passed the hidden text detection test
                                    - This is a positive indicator for document authenticity
                                    """)

                # Document Viewer with Annotations
                if has_text:
                    # Check if we have required libraries
                    if not HAS_PIL or not HAS_FLATTENING:
                        pass  # Skip viewer if libraries not available
                    else:
                        # Check if we have anything to annotate
                        has_hidden_text_result = False
                        hidden_text_items = []

                        # Check for hidden text detection results
                        if payload_exists and metadata and metadata.get("has_word_level"):
                            # Check if detection was already run (stored in session state)
                            if 'hidden_text_result' in st.session_state and st.session_state.get('last_pdf_path') == str(pdf_path):
                                result = st.session_state['hidden_text_result']
                                if result and not result.get('error'):
                                    hidden_text_items = result.get('hidden_items', [])
                                    has_hidden_text_result = len(hidden_text_items) > 0

                        # Check for Adobe/pikepdf payloads to identify rare properties
                        adobe_exists_viewer, adobe_path_viewer, _ = check_adobe_payload_exists(pdf_path)
                        pikepdf_exists_viewer, pikepdf_path_viewer, _ = check_pikepdf_payload_exists(pdf_path)

                        rare_adobe_props = None
                        rare_pikepdf_props = None
                        adobe_analysis_viewer = None
                        pikepdf_analysis_viewer = None

                        if adobe_exists_viewer or pikepdf_exists_viewer:
                            # Analyze and identify rare properties
                            if adobe_exists_viewer:
                                adobe_analysis_viewer = analyze_adobe_payload(adobe_path_viewer)
                                if adobe_analysis_viewer:
                                    rare_adobe_props = identify_rare_font_properties(adobe_analysis_viewer)

                        if pikepdf_exists_viewer:
                            pikepdf_analysis_viewer = analyze_adobe_payload(pikepdf_path_viewer)
                            if pikepdf_analysis_viewer:
                                rare_pikepdf_props = identify_rare_font_properties(pikepdf_analysis_viewer)

                        docling_table_cells = []
                        docling_alignment_items = []
                        docling_payload_data = None

                        if payload_exists and metadata and payload_path and payload_path.exists():
                            try:
                                with open(payload_path, 'r', encoding='utf-8') as docling_file:
                                    docling_payload_data = json.load(docling_file)
                            except Exception:
                                docling_payload_data = None

                        if docling_payload_data:
                            docling_table_cells = extract_table_cells_from_docling(docling_payload_data)
                            docling_alignment_items = detect_docling_alignment_anomalies(docling_payload_data)
                            colon_spacing_items = detect_colon_spacing_anomalies(docling_payload_data)

                            # Extract baselines for visualization (stored in payload by detect_docling_alignment_anomalies)
                            alignment_baselines_by_page = docling_payload_data.get("_alignment_baselines", {})

                            if docling_alignment_items and hidden_text_items:
                                hidden_boxes_by_page: Dict[int, List[Dict[str, float]]] = defaultdict(list)
                                for hidden_item in hidden_text_items:
                                    page_index = hidden_item.get('page')
                                    if page_index is None:
                                        continue

                                    for bbox_key in ('visible_bbox', 'hidden_bbox'):
                                        bbox_value = hidden_item.get(bbox_key)
                                        if not isinstance(bbox_value, dict):
                                            continue
                                        try:
                                            hx0, hy0, hx1, hy1 = extract_bbox_coords(bbox_value)
                                        except Exception:
                                            continue
                                        hidden_boxes_by_page[int(page_index)].append({
                                            "x0": hx0,
                                            "y0": hy0,
                                            "x1": hx1,
                                            "y1": hy1
                                        })

                                if hidden_boxes_by_page:
                                    filtered_alignment_items: List[Dict[str, Any]] = []
                                    for alignment_item in docling_alignment_items:
                                        bbox = alignment_item.get('bbox')
                                        page_index = alignment_item.get('page')
                                        if not bbox:
                                            continue

                                        overlaps_hidden = False
                                        if page_index in hidden_boxes_by_page:
                                            for hidden_bbox in hidden_boxes_by_page[page_index]:
                                                try:
                                                    if calculate_iou(bbox, hidden_bbox) > 0.05:
                                                        overlaps_hidden = True
                                                        break
                                                except Exception:
                                                    continue

                                        if not overlaps_hidden:
                                            filtered_alignment_items.append(alignment_item)

                                    docling_alignment_items = filtered_alignment_items

                        has_docling_table_overlays = len(docling_table_cells) > 0
                        has_docling_alignment_issues = len(docling_alignment_items) > 0
                        has_colon_spacing_patterns = len(colon_spacing_items) > 0

                        # Show viewer if we have anything to annotate
                        if has_hidden_text_result or rare_adobe_props or rare_pikepdf_props or has_docling_table_overlays or has_docling_alignment_issues or has_colon_spacing_patterns:
                            st.markdown("---")
                            st.subheader("📄 Document Viewer with Annotations")

                            # Get page count
                            try:
                                doc = fitz.open(pdf_path)
                                num_pages = len(doc)
                                doc.close()
                            except:
                                num_pages = 1

                            # Page selector
                            if num_pages > 1:
                                selected_page = st.selectbox(
                                    "Select page to view:",
                                    options=list(range(num_pages)),
                                    format_func=lambda x: f"Page {x + 1}",
                                    key="annotation_page_selector"
                                )
                            else:
                                selected_page = 0

                            # Render options
                            render_col1, render_col2 = st.columns([3, 1])
                            with render_col1:
                                st.caption("Annotations show hidden text and rare font properties")
                                if has_docling_table_overlays or has_docling_alignment_issues:
                                    st.caption("Additional overlays include Docling tables and alignment anomalies")
                                if has_colon_spacing_patterns:
                                    st.caption("Colon spacing patterns: Green=consistent, Red=deviation, Orange=right-aligned")
                            with render_col2:
                                dpi = st.selectbox("DPI:", options=[100, 150, 200, 300], index=1, key="render_dpi")

                            # Render the page with annotations
                            with st.spinner(f"Rendering page {selected_page + 1} with annotations..."):
                                # Get baselines for the selected page
                                page_baselines = None
                                if 'alignment_baselines_by_page' in locals() and alignment_baselines_by_page:
                                    page_baselines = alignment_baselines_by_page.get(selected_page)

                                annotated_img = render_pdf_with_annotations(
                                    pdf_path=pdf_path,
                                    page_num=selected_page,
                                    hidden_text_items=hidden_text_items if has_hidden_text_result else None,
                                    rare_adobe_properties=rare_adobe_props,
                                    rare_pikepdf_properties=rare_pikepdf_props,
                                    docling_table_cells=docling_table_cells if has_docling_table_overlays else None,
                                    alignment_anomalies=docling_alignment_items if has_docling_alignment_issues else None,
                                    alignment_baselines=page_baselines,
                                    colon_spacing_items=colon_spacing_items if has_colon_spacing_patterns else None,
                                    dpi=dpi
                                )

                                if annotated_img:
                                    # Display the annotated image
                                    st.image(annotated_img, caption=f"Page {selected_page + 1} with annotations", use_container_width=True)

                                    # Build detailed annotation information
                                    annotation_details = build_annotation_details(
                                        page_num=selected_page,
                                        hidden_text_items=hidden_text_items if has_hidden_text_result else None,
                                        rare_adobe_properties=rare_adobe_props,
                                        rare_pikepdf_properties=rare_pikepdf_props,
                                        adobe_analysis=adobe_analysis_viewer if adobe_exists_viewer else None,
                                        pikepdf_analysis=pikepdf_analysis_viewer if pikepdf_exists_viewer else None,
                                        docling_alignment_items=docling_alignment_items if has_docling_alignment_issues else None
                                    )

                                    # Display annotation details
                                    if annotation_details:
                                        st.markdown("---")
                                        st.markdown(f"#### Highlighted Items on Page {selected_page + 1}")
                                        st.caption(f"Found {len(annotation_details)} highlighted item(s) on this page")

                                        for idx, item in enumerate(annotation_details, 1):
                                            with st.expander(f"**Item {idx}: \"{item['text']}\"**", expanded=False):
                                                # Display text
                                                st.markdown(f"**Text:** `{item['text']}`")

                                                # Display color
                                                color_html = f'<div style="display: inline-block; width: 20px; height: 20px; background-color: {item["color"]}; border: 1px solid #000; vertical-align: middle; margin-right: 5px;"></div>'
                                                st.markdown(f"**Highlight Color:** {color_html} {item['color_name']}", unsafe_allow_html=True)

                                                # Display reasons
                                                st.markdown(f"**Reason(s):** {len(item['reasons'])} issue(s) detected")

                                                for reason_idx, reason in enumerate(item['reasons'], 1):
                                                    st.markdown(f"**Reason {reason_idx}:** {reason['type']}")
                                                    st.caption(f"   {reason['details']}")

                                                    # Show hidden text if this is an overlap
                                                    if 'hidden_text' in reason:
                                                        st.caption(f"   Hidden text beneath: `{reason['hidden_text']}`")
                                    else:
                                        st.info("No annotations on this page")
                                else:
                                    st.error("Failed to render page with annotations")


if __name__ == "__main__":
    main()
