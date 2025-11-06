"""
Batch Processing and Reporting for Document Forgery Detection

This script processes all PDFs in a directory through the complete pipeline:
1. Detect if PDF is OCRed or not
2. Generate payloads (Docling or Azure DI)
3. Generate Enhanced Text Info
4. Perform Cluster Analysis and Anomaly Detection
5. Detect Hidden/Overlapping Text (OCRed only)
6. Generate comprehensive reports

Usage:
    python batch_process_and_report.py <input_directory> [--output <custom_output_dir>]

    Default output: Document Pipeline/Reports/
"""

import sys
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import Counter
import numpy as np

# Import from pdf_ocr_detector
from pdf_ocr_detector import (
    check_pdf_has_text,
    check_pdf_restrictions,
    generate_docling_payload,
    generate_adobe_payload,
    generate_azure_di_payload,
    generate_enhanced_text_info,
    generate_enhanced_text_info_from_azure_di,
    perform_and_save_cluster_analysis,
    analyze_fonts_in_payload,
    analyze_adobe_payload,
    detect_hidden_text,
    get_anomaly_explanation,
    get_adobe_payload_path,
    PAYLOADS_PATH,
    ADOBE_PAYLOADS_PATH,
    AZURE_DI_PAYLOADS_PATH,
    ENHANCED_TEXT_INFO_PATH,
    CLUSTER_RESULTS_PATH,
    FEATURE_EXTRACTION_DPI
)


def process_single_pdf(pdf_path: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Process a single PDF through the complete pipeline.

    Returns:
        Dictionary with all results and metadata
    """
    result = {
        'filename': pdf_path.name,
        'success': False,
        'error': None,
        'is_ocred': None,
        'is_restricted': None,
        'payload_path': None,
        'adobe_payload_path': None,
        'enhanced_text_info_path': None,
        'cluster_result_path': None,
        'hidden_text_result': None,
        'pipeline': None  # 'docling' or 'azure_di'
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*80}")

    try:
        # Step 1: Check if PDF has text (OCRed check)
        if verbose:
            print("\n[1/6] Checking if PDF is OCRed...")

        has_text, total_pages, text_pages, _ = check_pdf_has_text(pdf_path)
        result['is_ocred'] = has_text
        result['total_pages'] = total_pages
        result['text_pages'] = text_pages

        if verbose:
            if has_text:
                print(f"[OK] PDF is OCRed ({text_pages}/{total_pages} pages with text)")
            else:
                print(f"[OK] PDF is NOT OCRed (will use Azure DI pipeline)")

        # Step 2: Check restrictions (OCRed only)
        if has_text:
            if verbose:
                print("\n[2/6] Checking PDF restrictions...")

            restrictions = check_pdf_restrictions(pdf_path)
            result['is_restricted'] = restrictions.get('has_restrictions', False)
            result['restrictions'] = restrictions

            if verbose:
                if result['is_restricted']:
                    print(f"[WARNING] PDF has restrictions: {restrictions}")
                else:
                    print("[OK] PDF has no restrictions")

        # Step 3: Generate payload
        if verbose:
            print(f"\n[3/6] Generating payload...")

        if has_text:
            # OCRed: Use Docling + Adobe
            result['pipeline'] = 'docling'
            success, message, payload_path = generate_docling_payload(pdf_path, force_regenerate=False)

            if not success:
                result['error'] = f"Docling payload failed: {message}"
                return result

            result['payload_path'] = payload_path
            if verbose:
                print(f"[OK] Docling payload: {payload_path.name}")

            # Generate Adobe payload for font analysis
            if verbose:
                print(f"[3b/6] Generating Adobe payload...")

            adobe_success, adobe_message, adobe_payload_path = generate_adobe_payload(pdf_path, force_regenerate=False)

            if adobe_success:
                result['adobe_payload_path'] = adobe_payload_path
                if verbose:
                    print(f"[OK] Adobe payload: {adobe_payload_path.name}")
            else:
                if verbose:
                    print(f"[WARNING] Adobe payload generation failed: {adobe_message}")
                # Don't fail the entire process if Adobe fails - continue without it
                result['adobe_payload_path'] = None
        else:
            # Not OCRed: Use Azure DI
            result['pipeline'] = 'azure_di'
            success, message, payload_path = generate_azure_di_payload(pdf_path, force_regenerate=False)

            if not success:
                result['error'] = f"Azure DI payload failed: {message}"
                return result

            result['payload_path'] = payload_path
            if verbose:
                print(f"[OK] Azure DI payload: {payload_path.name}")

        # Step 4: Generate Enhanced Text Info
        if verbose:
            print(f"\n[4/6] Generating Enhanced Text Info (DPI: {FEATURE_EXTRACTION_DPI})...")

        # Force CPU mode for batch processing to avoid CUDA context corruption
        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Temporarily disable CUDA

        try:
            if has_text:
                success, message, enhanced_path = generate_enhanced_text_info(pdf_path, force_regenerate=False)
            else:
                success, message, enhanced_path = generate_enhanced_text_info_from_azure_di(pdf_path, force_regenerate=False)
        finally:
            # Restore CUDA visibility
            if old_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)

        if not success:
            result['error'] = f"Enhanced text info failed: {message}"
            return result

        result['enhanced_text_info_path'] = enhanced_path
        if verbose:
            print(f"[OK] Enhanced text info: {enhanced_path.name}")

        # Step 5: Perform Cluster Analysis
        if verbose:
            print(f"\n[5/6] Performing Cluster Analysis...")

        success, message, cluster_result = perform_and_save_cluster_analysis(
            enhanced_text_info_path=enhanced_path,
            use_gpu=False,  # Use CPU for batch processing to avoid CUDA errors
            max_k=4
        )

        if not success:
            result['error'] = f"Cluster analysis failed: {message}"
            return result

        # Find cluster result path
        cluster_result_path = CLUSTER_RESULTS_PATH / f"{enhanced_path.stem}_cluster_result.pkl"
        result['cluster_result_path'] = cluster_result_path
        result['cluster_result'] = cluster_result

        if verbose:
            print(f"[OK] Cluster analysis: {cluster_result_path.name}")
            print(f"  - K-means clusters: {cluster_result.kmeans_n_clusters}")
            print(f"  - DBSCAN clusters: {cluster_result.dbscan_n_clusters}")
            if hasattr(cluster_result, 'ensemble_anomaly_flags'):
                n_anomalies = int(np.sum(cluster_result.ensemble_anomaly_flags))
                print(f"  - Anomalies detected: {n_anomalies}")

        # Step 6: Detect Hidden Text (OCRed only)
        if has_text and result['payload_path']:
            if verbose:
                print(f"\n[6/6] Detecting Hidden/Overlapping Text...")

            try:
                hidden_text_result = detect_hidden_text(pdf_path, result['payload_path'])
                result['hidden_text_result'] = hidden_text_result

                if hidden_text_result and verbose:
                    n_hidden = len(hidden_text_result.get('hidden_cells', []))
                    if n_hidden > 0:
                        print(f"[WARNING] Found {n_hidden} hidden/overlapping text elements!")
                    else:
                        print(f"[OK] No hidden text detected")
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Hidden text detection failed: {str(e)}")
                result['hidden_text_result'] = None

        result['success'] = True

        if verbose:
            print(f"\n{'='*80}")
            print(f"[OK] Processing complete for {pdf_path.name}")
            print(f"{'='*80}")

        return result

    except Exception as e:
        import traceback
        result['error'] = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(f"\n[ERROR] Error processing {pdf_path.name}: {str(e)}")
            traceback.print_exc()
        return result


def generate_report(result: Dict[str, Any], pdf_path: Path) -> str:
    """
    Generate a comprehensive markdown report for a single PDF.

    Args:
        result: Processing result from process_single_pdf()
        pdf_path: Path to the PDF file

    Returns:
        Markdown formatted report string
    """
    report_lines = []

    # Header
    report_lines.append(f"# Document Analysis Report")
    report_lines.append(f"**File**: {result['filename']}")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**DPI**: {FEATURE_EXTRACTION_DPI}")
    report_lines.append("")

    if not result['success']:
        report_lines.append(f"## [ERROR] Processing Failed")
        report_lines.append(f"**Error**: {result['error']}")
        return "\n".join(report_lines)

    # Document Type
    report_lines.append(f"## 📄 Document Type")
    report_lines.append(f"- **OCRed**: {'Yes' if result['is_ocred'] else 'No'}")
    report_lines.append(f"- **Pipeline**: {result['pipeline'].upper()}")

    if result['is_ocred'] and result['is_restricted'] is not None:
        report_lines.append(f"- **Restricted**: {'Yes' if result['is_restricted'] else 'No'}")
        if result['is_restricted'] and 'restrictions' in result:
            restrictions = result['restrictions']
            report_lines.append(f"  - Print: {'Allowed' if restrictions.get('print_allowed', False) else 'Restricted'}")
            report_lines.append(f"  - Copy: {'Allowed' if restrictions.get('copy_allowed', False) else 'Restricted'}")
            report_lines.append(f"  - Modify: {'Allowed' if restrictions.get('modify_allowed', False) else 'Restricted'}")

    report_lines.append("")

    # Font Analysis (OCRed only)
    if result['is_ocred']:
        report_lines.append(f"## 🔤 Font Analysis")

        try:
            # Get payload path - from result or construct from pdf_path
            payload_path = result.get('payload_path')
            if not payload_path:
                payload_path = PAYLOADS_PATH / f"{pdf_path.stem}_Docling_raw_payload.json"

            if payload_path and payload_path.exists():
                font_analysis = analyze_fonts_in_payload(payload_path)
            else:
                font_analysis = None

            if font_analysis and 'font_combinations' in font_analysis:
                font_combinations = font_analysis['font_combinations']

                # Calculate total items
                total_items = sum(data['count'] for data in font_combinations.values())

                # Find uncommon fonts (less than 5% of total or less than 5 items)
                uncommon_threshold = max(5, total_items * 0.05)
                uncommon_fonts = {
                    font: data for font, data in font_combinations.items()
                    if data['count'] < uncommon_threshold
                }

                if uncommon_fonts:
                    report_lines.append(f"**Uncommon font combinations found** ({len(uncommon_fonts)} combinations):")
                    report_lines.append("")

                    # Sort by count (rarest first)
                    sorted_uncommon = sorted(uncommon_fonts.items(), key=lambda x: x[1]['count'])

                    for font, data in sorted_uncommon:
                        count = data['count']
                        percentage = (count / total_items) * 100
                        report_lines.append(f"### {font}")
                        report_lines.append(f"- Count: {count} ({percentage:.1f}%)")
                        report_lines.append(f"- Example texts:")
                        for text in data['texts'][:5]:  # Show up to 5 examples
                            report_lines.append(f"  - \"{text}\"")
                        if len(data['texts']) > 5:
                            report_lines.append(f"  - ... and {len(data['texts']) - 5} more")
                        report_lines.append("")
                else:
                    report_lines.append(f"**All fonts are commonly used** (no uncommon combinations detected)")
                    report_lines.append(f"- Total font combinations: {len(font_combinations)}")
                    report_lines.append(f"- All combinations appear frequently (>{uncommon_threshold:.0f} items or >5%)")
                    report_lines.append("")
            else:
                report_lines.append("*Font analysis data not available*")
                report_lines.append("")

        except Exception as e:
            report_lines.append(f"*Font analysis failed: {str(e)}*")
            report_lines.append("")

    # Adobe Attributes (OCRed only)
    if result['is_ocred']:
        report_lines.append(f"## 📐 Adobe Attributes")

        try:
            # Get Adobe payload path - from result or construct from pdf_path
            adobe_payload_path = result.get('adobe_payload_path')
            if not adobe_payload_path:
                adobe_payload_path = get_adobe_payload_path(pdf_path)

            if adobe_payload_path and adobe_payload_path.exists():
                adobe_analysis = analyze_adobe_payload(adobe_payload_path)

                if adobe_analysis:
                    # Check if attributes are diverse or uniform
                    attributes_to_check = [
                        ('width_category', 'Width'),
                        ('height_category', 'Height'),
                        ('aspect_ratio_category', 'Aspect Ratio'),
                        ('x0_category', 'X Position'),
                        ('y0_category', 'Y Position')
                    ]

                    uniform_attributes = []
                    diverse_attributes = []

                    for attr_key, attr_name in attributes_to_check:
                        if attr_key in adobe_analysis:
                            categories = adobe_analysis[attr_key]
                            if len(categories) == 1:
                                uniform_attributes.append((attr_name, list(categories.keys())[0]))
                            else:
                                diverse_attributes.append((attr_name, categories))

                    if uniform_attributes:
                        report_lines.append(f"**Uniform attributes** (all items in same category):")
                        for attr_name, category in uniform_attributes:
                            report_lines.append(f"- **{attr_name}**: All items are `{category}`")
                        report_lines.append("")

                    if diverse_attributes:
                        report_lines.append(f"**Diverse attributes** (multiple categories found):")
                        report_lines.append("")

                        for attr_name, categories in diverse_attributes:
                            report_lines.append(f"### {attr_name}")
                            # Sort by count to find uncommon ones
                            total = sum(data['count'] for data in categories.values())
                            uncommon_threshold = max(5, total * 0.05)

                            uncommon_categories = {
                                cat: data for cat, data in categories.items()
                                if data['count'] < uncommon_threshold
                            }

                            if uncommon_categories:
                                report_lines.append(f"**Uncommon categories**:")
                                for cat, data in sorted(uncommon_categories.items(), key=lambda x: x[1]['count']):
                                    count = data['count']
                                    percentage = (count / total) * 100
                                    report_lines.append(f"- **{cat}**: {count} items ({percentage:.1f}%)")
                                    report_lines.append(f"  - Examples: {', '.join([f'"{t}"' for t in data['texts'][:3]])}")
                            else:
                                report_lines.append(f"*All categories are commonly used*")
                            report_lines.append("")
                else:
                    report_lines.append("*Adobe attributes not available*")
                    report_lines.append("")
            else:
                report_lines.append("*Adobe payload not found (may not have been generated)*")
                report_lines.append("")

        except Exception as e:
            report_lines.append(f"*Adobe attribute analysis failed: {str(e)}*")
            report_lines.append("")

    # Hidden Text Detection (OCRed only)
    if result['is_ocred'] and 'hidden_text_result' in result and result['hidden_text_result']:
        hidden_text = result['hidden_text_result']

        report_lines.append(f"## 🔎 Hidden/Overlapping Text Detection")
        report_lines.append("")

        hidden_cells = hidden_text.get('hidden_cells', [])
        n_hidden = len(hidden_cells)

        if n_hidden > 0:
            report_lines.append(f"[WARNING] **{n_hidden} hidden or overlapping text elements detected!**")
            report_lines.append("")
            report_lines.append("These text elements exist in the PDF but are not visible in the rendered output:")
            report_lines.append("")

            for i, cell in enumerate(hidden_cells[:20], 1):  # Top 20
                text = cell.get('text', '(no text)')
                match_status = cell.get('match_status', 'unknown')
                iou = cell.get('iou', 0)

                report_lines.append(f"### Hidden Element #{i}")
                report_lines.append(f"**Text**: \"{text}\"")
                report_lines.append(f"- **Match Status**: {match_status}")
                if iou > 0:
                    report_lines.append(f"- **Overlap (IOU)**: {iou:.2%}")
                report_lines.append("")

            if n_hidden > 20:
                report_lines.append(f"*... and {n_hidden - 20} more hidden elements*")
                report_lines.append("")
        else:
            report_lines.append("[OK] **No hidden or overlapping text detected**")
            report_lines.append("")

    # Anomaly Detection
    if 'cluster_result' in result and result['cluster_result']:
        cluster_result = result['cluster_result']

        if hasattr(cluster_result, 'ensemble_anomaly_flags'):
            report_lines.append(f"## 🔍 Anomaly Detection")
            report_lines.append(f"*Contamination: 3% | Feature threshold: >2.5σ | Ensemble: 2+/3 methods*")
            report_lines.append("")

            n_anomalies = int(np.sum(cluster_result.ensemble_anomaly_flags))
            anomaly_rate = float(np.mean(cluster_result.ensemble_anomaly_flags)) * 100
            high_confidence = int(np.sum(cluster_result.ensemble_agreement_count == 3))
            medium_confidence = int(np.sum(cluster_result.ensemble_agreement_count == 2))

            report_lines.append(f"- **Total Anomalies**: {n_anomalies} ({anomaly_rate:.1f}%)")
            report_lines.append(f"- **High Confidence** (3/3 methods): {high_confidence}")
            report_lines.append(f"- **Medium Confidence** (2/3 methods): {medium_confidence}")
            report_lines.append("")

            if n_anomalies > 0:
                # Get anomaly indices
                anomaly_indices = np.where(cluster_result.ensemble_anomaly_flags)[0]

                # Sort by confidence
                sorted_indices = sorted(
                    anomaly_indices,
                    key=lambda idx: cluster_result.ensemble_agreement_count[idx],
                    reverse=True
                )

                report_lines.append(f"### Detected Anomalies")
                report_lines.append("")

                for rank, idx in enumerate(sorted_indices[:20], 1):  # Top 20
                    agreement = cluster_result.ensemble_agreement_count[idx]
                    text = cluster_result.texts[idx]

                    confidence_level = "HIGH" if agreement == 3 else "MEDIUM"
                    confidence_badge = "🔴" if agreement == 3 else "🟡"

                    report_lines.append(f"#### {confidence_badge} Anomaly #{rank} - {confidence_level} Confidence ({agreement}/3)")
                    report_lines.append(f"**Text**: \"{text}\"")
                    report_lines.append("")

                    # Get features
                    features = cluster_result.features_standardized[idx]

                    # Find anomalous features (>2.5σ)
                    anomalous_features = []
                    for feat_idx, feat_name in enumerate(cluster_result.feature_names):
                        feat_value = features[feat_idx]
                        if abs(feat_value) > 2.5:
                            anomalous_features.append((feat_name, feat_value))

                    if anomalous_features:
                        report_lines.append("**Why it's flagged**:")
                        # Sort by magnitude
                        sorted_features = sorted(anomalous_features, key=lambda x: abs(x[1]), reverse=True)

                        for feat_name, feat_value in sorted_features[:3]:  # Top 3 features
                            explanation = get_anomaly_explanation(feat_name, feat_value)
                            report_lines.append(f"- **{feat_name}** ({feat_value:+.2f}σ): {explanation}")
                    else:
                        report_lines.append("**Why it's flagged**: General pattern deviation (no single feature >2.5σ)")

                    report_lines.append("")

                if n_anomalies > 20:
                    report_lines.append(f"*... and {n_anomalies - 20} more anomalies (see cluster result file for full details)*")
                    report_lines.append("")
            else:
                report_lines.append("[OK] **No significant anomalies detected**")
                report_lines.append("")

    # Cluster Analysis - Smallest Clusters
    if 'cluster_result' in result and result['cluster_result']:
        cluster_result = result['cluster_result']

        report_lines.append(f"## 📊 Cluster Analysis - Smallest Clusters")
        report_lines.append("")

        # Use DBSCAN clusters
        labels = cluster_result.dbscan_labels
        unique_labels = sorted(set(labels))

        # Get cluster sizes
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:  # Skip noise
                size = np.sum(labels == label)
                cluster_sizes.append((label, size))

        # Sort by size (smallest first)
        cluster_sizes.sort(key=lambda x: x[1])

        report_lines.append(f"**DBSCAN Clusters**: {len(cluster_sizes)} clusters + {np.sum(labels == -1)} noise points")
        report_lines.append("")

        # Report on smallest 3 clusters
        for label, size in cluster_sizes[:3]:
            report_lines.append(f"### Cluster {label} ({size} items)")

            # Get items in this cluster
            cluster_mask = labels == label
            cluster_texts = [text for i, text in enumerate(cluster_result.texts) if cluster_mask[i]]
            cluster_features = cluster_result.features_standardized[cluster_mask]

            # Get feature importance for this cluster
            feature_importance = cluster_result.get_feature_importance_for_separation(labels, label)

            report_lines.append(f"**Items in cluster**:")
            for i, text in enumerate(cluster_texts[:10], 1):  # Show up to 10
                report_lines.append(f"{i}. \"{text}\"")

            if len(cluster_texts) > 10:
                report_lines.append(f"... and {len(cluster_texts) - 10} more items")

            report_lines.append("")
            report_lines.append(f"**Why these items cluster together**:")

            # feature_importance is already a sorted list of tuples: [(feat_name, abs_diff, percentage), ...]
            for feat_name, abs_diff, percentage in feature_importance[:3]:
                # Get mean value for this cluster
                feat_idx = cluster_result.feature_names.index(feat_name)
                cluster_mean = np.mean(cluster_features[:, feat_idx])

                # Get human explanation
                explanation = get_anomaly_explanation(feat_name, cluster_mean)

                report_lines.append(f"- **{feat_name}** ({percentage:.1f}% contribution)")
                report_lines.append(f"  - Cluster mean: {cluster_mean:+.2f}σ")
                report_lines.append(f"  - {explanation}")

            report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append(f"*Report generated by Document Forgery Detection Pipeline*")
    report_lines.append(f"*Cluster result: `{result['cluster_result_path'].name if result.get('cluster_result_path') else 'N/A'}`*")

    return "\n".join(report_lines)


def batch_process_directory(input_dir: Path, output_dir: Path, verbose: bool = True):
    """
    Process all PDFs in a directory and generate reports.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save reports
        verbose: Print progress messages
    """
    # Find all PDFs
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    print(f"Reports will be saved to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each PDF
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")

        try:
            # Process PDF
            result = process_single_pdf(pdf_path, verbose=verbose)
            results.append(result)

            # Generate report
            report_content = generate_report(result, pdf_path)

            # Save report
            report_filename = f"{pdf_path.stem}_report.md"
            report_path = output_dir / report_filename

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            print(f"[OK] Report saved: {report_filename}")

        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to process {pdf_path.name}: {str(e)}")
            traceback.print_exc()

            # Save error report
            error_report = f"# Error Report\n\n**File**: {pdf_path.name}\n\n**Error**: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            report_filename = f"{pdf_path.stem}_ERROR_report.md"
            report_path = output_dir / report_filename

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(error_report)

    # Generate summary report
    print(f"\n{'='*80}")
    print("Generating summary report...")

    summary_lines = []
    summary_lines.append(f"# Batch Processing Summary Report")
    summary_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"**Input Directory**: {input_dir}")
    summary_lines.append(f"**Total Files**: {len(pdf_files)}")
    summary_lines.append("")

    # Count successes and failures
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    summary_lines.append(f"## Processing Results")
    summary_lines.append(f"- **Successful**: {successful}")
    summary_lines.append(f"- **Failed**: {failed}")
    summary_lines.append("")

    # Count by pipeline
    ocred_count = sum(1 for r in results if r['is_ocred'])
    not_ocred_count = len(results) - ocred_count

    summary_lines.append(f"## Document Types")
    summary_lines.append(f"- **OCRed**: {ocred_count}")
    summary_lines.append(f"- **Not OCRed**: {not_ocred_count}")
    summary_lines.append("")

    # Anomaly statistics
    total_anomalies = 0
    files_with_anomalies = 0

    for r in results:
        if r['success'] and 'cluster_result' in r and r['cluster_result']:
            cluster_result = r['cluster_result']
            if hasattr(cluster_result, 'ensemble_anomaly_flags'):
                n_anomalies = int(np.sum(cluster_result.ensemble_anomaly_flags))
                total_anomalies += n_anomalies
                if n_anomalies > 0:
                    files_with_anomalies += 1

    summary_lines.append(f"## Anomaly Detection Summary")
    summary_lines.append(f"- **Total Anomalies Detected**: {total_anomalies}")
    summary_lines.append(f"- **Files with Anomalies**: {files_with_anomalies}/{successful}")
    summary_lines.append("")

    # List all files with their status
    summary_lines.append(f"## File Details")
    summary_lines.append("")

    for r in results:
        status = "[OK]" if r['success'] else "[ERROR]"
        ocred_status = "OCRed" if r['is_ocred'] else "Not OCRed"

        summary_lines.append(f"### {status} {r['filename']}")
        summary_lines.append(f"- **Type**: {ocred_status}")

        if r['success']:
            summary_lines.append(f"- **Pipeline**: {r['pipeline'].upper()}")
            if 'cluster_result' in r and r['cluster_result']:
                cluster_result = r['cluster_result']
                if hasattr(cluster_result, 'ensemble_anomaly_flags'):
                    n_anomalies = int(np.sum(cluster_result.ensemble_anomaly_flags))
                    summary_lines.append(f"- **Anomalies**: {n_anomalies}")
        else:
            summary_lines.append(f"- **Error**: {r['error']}")

        summary_lines.append("")

    # Save summary report
    summary_path = output_dir / "_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print(f"[OK] Summary report saved: {summary_path.name}")
    print(f"{'='*80}")


def main():
    """Main entry point for batch processing."""
    if len(sys.argv) < 2:
        print("Usage: python batch_process_and_report.py <input_directory> [--output <output_directory>]")
        print("\nExample:")
        print("  python batch_process_and_report.py 'C:\\PDFs' --output 'C:\\Reports'")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Parse output directory
    # Default: Save to "Document Pipeline/Reports/" folder
    script_dir = Path(__file__).parent
    output_dir = script_dir / "Reports"

    if '--output' in sys.argv:
        output_idx = sys.argv.index('--output')
        if output_idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[output_idx + 1])

    print(f"\n{'='*80}")
    print(f"Document Forgery Detection - Batch Processing")
    print(f"{'='*80}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Feature Extraction DPI: {FEATURE_EXTRACTION_DPI}")
    print(f"{'='*80}\n")

    batch_process_directory(input_dir, output_dir, verbose=True)


if __name__ == "__main__":
    main()
