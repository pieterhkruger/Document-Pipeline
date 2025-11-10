Scope & Audience

Delivers PDF OCR, restriction, and payload intelligence for document-fraud investigations across the Document Pipeline (pdf_ocr_detector.py (line 1)).
Streamlit UI with emoji feedback serves primary analysts for rapid triage and review (pdf_ocr_detector.py (line 5385), pdf_ocr_detector.py (line 5434)).
Enriched payload outputs support ML/data engineers who need word-level features and metadata for downstream models (pdf_ocr_detector.py (line 1138)).
Batch automation paths target ops and QA teams processing large corpora and compiling reports (batch_process_and_report.py (line 736)).
Operational Modes

Interactive runs load PDFs from Actual Use Cases/ and surface live metrics through the Streamlit front end (pdf_ocr_detector.py (line 191), pdf_ocr_detector.py (line 5403)).
Batch CLI ingests directories, executes the full pipeline, and persists consolidated reports under Reports/ (batch_process_and_report.py (line 52), batch_process_and_report.py (line 247)).
Service toggles gate Adobe, Azure, and Google integrations with runtime capability checks and fallbacks (pdf_ocr_detector.py (line 204), pdf_ocr_detector.py (line 335), pdf_ocr_detector.py (line 379)).
Hidden-text mode performs 400 DPI rasterization with Tesseract/Poppler auto-discovery before OCR comparison (pdf_ocr_detector.py (line 72), pdf_ocr_detector.py (line 2809)).
Core Analysis Features

OCR inspection counts text-bearing pages, previews content, and tracks coverage percentages (pdf_ocr_detector.py (line 914)).
Security audit reports encryption status and permission flags via pikepdf with PyMuPDF fallback (pdf_ocr_detector.py (line 946)).
Docling payload generation caches timestamped JSON with cells, fonts, colors, and confidence statistics (pdf_ocr_detector.py (line 1049), pdf_ocr_detector.py (line 1138)).
Font analytics summarize usage distributions and cell statistics for drill-down review (pdf_ocr_detector.py (line 1298), pdf_ocr_detector.py (line 3010)).
Adobe/pikepdf payload readers surface embedded font traits and highlight rare property combinations (pdf_ocr_detector.py (line 1427), pdf_ocr_detector.py (line 4755)).
Advanced & Optional Modules

Hidden-text detection aligns original vs flattened payloads and flags overlapping or missing spans (pdf_ocr_detector.py (line 2809), pdf_ocr_detector.py (line 4299)).
Azure Document Intelligence pipeline normalizes SDK responses for non-OCR PDFs and persists raw payloads (azure_document_intelligence.py (line 112), azure_document_intelligence.py (line 246)).
Enhanced text info augments every cell with color clustering and gradient/ELA metrics (pdf_ocr_detector.py (line 3689), pillow_color_clustering.py (line 231), edge_gradient_analysis.py (line 492)).
Cluster analysis produces anomaly scores plus human-readable explanations for suspicious features (pdf_ocr_detector.py (line 4154), cluster_analysis.py (line 759), cluster_analysis.py (line 225)).
Layout diagnostics capture colon spacing, alignment baselines, and overlapping bbox anomalies for fraud cues (pdf_ocr_detector.py (line 2177), pdf_ocr_detector.py (line 2289), pdf_ocr_detector.py (line 2688)).
Data & Storage Requirements

Source PDFs must reside in the project’s Actual Use Cases/ directory by default (pdf_ocr_detector.py (line 191)).
Enhanced Docling payloads save under Docling payloads/ using timestamped naming (pdf_ocr_detector.py (line 195), pdf_ocr_detector.py (line 1040)).
Additional outputs populate sibling folders for Adobe, pikepdf, Azure DI, enhanced text info, cluster results, and overlapping bboxes (pdf_ocr_detector.py (line 196), pdf_ocr_detector.py (line 199), pdf_ocr_detector.py (line 200), pdf_ocr_detector.py (line 201)).
Unified raw API responses are written to ../Glyph_anomalies/raw_payloads, which must exist and be writeable (pdf_ocr_detector.py (line 193)).
Configuration & Environment

.env presence is validated before enabling external APIs; missing keys disable corresponding integrations (pdf_ocr_detector.py (line 297), pdf_ocr_detector.py (line 335), pdf_ocr_detector.py (line 379)).
.env_example lists required credentials for Azure, Adobe, Google, AWS, DocParser, and other services (.env_example:1).
Feature switches USE_ADOBE_API_IF_AVAILABLE, USE_AZURE_DI_API_IF_AVAILABLE, and USE_GOOGLE_API_IF_AVAILABLE define runtime behavior (pdf_ocr_detector.py (line 204), pdf_ocr_detector.py (line 205), pdf_ocr_detector.py (line 206)).
Tesseract helpers locate binaries and set TESSDATA_PREFIX, ensuring hidden-text detection works across OS installs (pdf_ocr_detector.py (line 72), pdf_ocr_detector.py (line 87)).
GPU and high-DPI workflows rely on Poppler, PyTorch, and optional CUDA; availability is detected at runtime (pdf_ocr_detector.py (line 212), pillow_color_clustering.py (line 66), pytorch_kmeans.py (line 173)).
External Dependencies

Core stack requires Streamlit, PyMuPDF, pikepdf, and Docling (pdf_ocr_detector.py (line 16)).
Hidden-text pipeline depends on pdf2image, pytesseract, Pillow, and Poppler utilities (pdf_ocr_detector.py (line 44), pdf_ocr_detector.py (line 2809)).
Feature extraction uses numpy, scipy, scikit-learn, and torch for clustering and statistics (cluster_analysis.py (line 225), cluster_analysis.py (line 594)).
Azure Document Intelligence SDK provides cloud OCR support when credentials are supplied (azure_document_intelligence.py (line 45)).
Optional Adobe runner is loaded from Unit testing - Tools/tools/adobe/runner.py when available (pdf_ocr_detector.py (line 214)).
Non-Functional Expectations

Payload caches avoid unnecessary recomputation unless users request regeneration (pdf_ocr_detector.py (line 1049), pdf_ocr_detector.py (line 1138)).
Fallback logic gracefully downgrades to local parsing when cloud services or runners are missing (pdf_ocr_detector.py (line 3389), pdf_ocr_detector.py (line 3547)).
Hidden-text rasterization defaults to 400 DPI; tuning may be needed to balance speed vs fidelity for large PDFs (pdf_ocr_detector.py (line 212), pdf_ocr_detector.py (line 2809)).
UI maintains responsiveness with spinners, metrics, and emoji alerts to signal status (pdf_ocr_detector.py (line 5434), pdf_ocr_detector.py (line 5468)).
GPU acceleration is optional but expected for timely clustering; CPU fallbacks must stay reliable (pillow_color_clustering.py (line 66), pytorch_kmeans.py (line 144)).
Operational Considerations

Batch reporting writes JSON summaries and optional pickle caches; monitor disk growth over long runs (batch_process_and_report.py (line 247), batch_process_and_report.py (line 600)).
Temporary artifacts from hidden-text detection remain suppressed unless KEEP_INTERMEDIATE_FILES is toggled (pdf_ocr_detector.py (line 213), pdf_ocr_detector.py (line 2809)).
Overlapping bbox detection feeds annotation rendering for investigators to visualize suspect regions (pdf_ocr_detector.py (line 2688), pdf_ocr_detector.py (line 4822)).
Annotation rendering depends on PIL font availability; deployment images should bundle the required fonts (pdf_ocr_detector.py (line 4822)).
Open Questions & Gaps

README references requirements-313.txt, but the file is absent; confirm dependency pinning strategy (README.md (line 49)).
RAW_PAYLOADS_PATH targets a parent directory outside the repo; clarify provisioning and permissions before deployment (pdf_ocr_detector.py (line 193)).
Google Document AI checks exist without a concrete execution path; decide whether to complete or remove this integration (pdf_ocr_detector.py (line 206), pdf_ocr_detector.py (line 379)).
Emoji-rich UI assumes UTF-8 capable environments; verify font support for targeted deployments (pdf_ocr_detector.py (line 5434), pdf_ocr_detector.py (line 5465)).