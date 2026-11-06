#!/usr/bin/env python3
"""
ARIA Pipeline Evaluation - BATCH MODE with Continuous PDF Updates (FIXED)
Runs evaluation in batches and updates PDF report after each batch completes
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

# Install: pip install reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Import ARIA pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph.orchestrator import run_pipeline

# ============================================================================
# STEP 1: COLLECT METRICS
# ============================================================================
class MetricsCollector:
    """Collects metrics while running pipeline"""

    def __init__(self):
        self.total_images = 0
        self.successful = 0
        self.failed = 0
        self.times = []
        self.routes = defaultdict(int)
        self.errors = defaultdict(int)
        self.results = []

    def add_result(self, image_name, success, exec_time, route=None, error=None):
        """Record one image result"""
        self.total_images += 1
        self.times.append(exec_time)

        if success:
            self.successful += 1
            if route:
                self.routes[route] += 1
        else:
            self.failed += 1
            if error:
                error_type = error.split(':')[0]
                self.errors[error_type] += 1

        self.results.append({
            'image': image_name,
            'status': 'success' if success else 'failed',
            'time': exec_time,
            'route': route,
            'error': error
        })

    def get_stats(self):
        """Get all statistics"""
        success_rate = (self.successful / self.total_images * 100) if self.total_images > 0 else 0

        return {
            'total_images': self.total_images,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': round(success_rate, 2),
            'total_time_seconds': round(sum(self.times), 2),
            'avg_time_seconds': round(statistics.mean(self.times), 3) if self.times else 0,
            'median_time_seconds': round(statistics.median(self.times), 3) if self.times else 0,
            'min_time_seconds': round(min(self.times), 3) if self.times else 0,
            'max_time_seconds': round(max(self.times), 3) if self.times else 0,
            'routes': dict(self.routes),
            'errors': dict(self.errors),
        }

# ============================================================================
# STEP 2: LOAD IMAGES
# ============================================================================
def load_images(dataset_path, limit=None):
    """Load all RGB images from OCID dataset"""
    path = Path(dataset_path)
    images = sorted(list(path.rglob("rgb/*.png")))

    if limit:
        images = images[:limit]

    logger.info(f"Found {len(images)} images")
    return images

# ============================================================================
# STEP 3: LOAD CHECKPOINT (RESUME FROM WHERE WE LEFT OFF) - FIXED
# ============================================================================
def load_checkpoint(checkpoint_file):
    """Load previous progress - FIXED with better error handling"""
    if not Path(checkpoint_file).exists():
        logger.info(f"No checkpoint found at: {checkpoint_file}")
        return None, None

    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        # Validate checkpoint structure
        if 'processed' not in checkpoint or 'metrics' not in checkpoint:
            logger.warning(f"Checkpoint missing required keys")
            return None, None

        processed = checkpoint['processed']
        metrics = checkpoint['metrics']

        logger.info(f"✓ Loaded checkpoint: {len(processed)} images already done")
        return processed, metrics

    except json.JSONDecodeError as e:
        logger.warning(f"Checkpoint JSON corrupted: {e}")
        return None, None
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return None, None

def save_checkpoint(checkpoint_file, processed_images, metrics):
    """Save progress for resuming later"""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'processed': processed_images,
        'metrics': metrics
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# ============================================================================
# STEP 4: RUN PIPELINE ON EACH IMAGE
# ============================================================================
def run_pipeline_on_image(image_path):
    """Run ARIA pipeline on one image, return (success, time, route, error)"""
    start = time.time()

    try:
        # Create state for pipeline
        state = {
            "user_input": f"Analyze image: {Path(image_path).name}",
            "session_id": f"eval-{int(time.time())}",
            "image_path": str(image_path),
            "video_path": None,
            "retry_count": 0,
        }

        # Run pipeline
        result = run_pipeline(state)

        # Get route
        route = result.get('route', 'unknown')

        elapsed = time.time() - start
        logger.info(f"✓ {Path(image_path).name} ({route}) - {elapsed:.2f}s")

        return True, elapsed, route, None

    except Exception as e:
        elapsed = time.time() - start
        error = f"{type(e).__name__}: {str(e)}"
        logger.error(f"✗ {Path(image_path).name} - {error}")
        return False, elapsed, None, error

# ============================================================================
# STEP 5: GENERATE PDF REPORT (WITH BATCH INFO)
# ============================================================================
def generate_pdf_report(output_file, metrics, total_images_in_dataset, batch_number=None, batch_size=None):
    """Generate professional PDF report with batch information"""
    logger.info(f"Generating PDF report: {output_file}")

    doc = SimpleDocTemplate(output_file, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("ARIA Pipeline Evaluation Report", styles['Heading1']))

    # Subtitle with batch info
    if batch_number and batch_size:
        subtitle = f"OCID-Ref Dataset | Batch {batch_number} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        status = f"(Batch in progress: {metrics.total_images}/{total_images_in_dataset} images processed)"
    else:
        subtitle = f"OCID-Ref Dataset | Final Report | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        status = f"(Evaluation complete: {metrics.total_images}/{total_images_in_dataset} images processed)"

    story.append(Paragraph(subtitle, styles['Heading2']))
    story.append(Paragraph(status, styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    story.append(PageBreak())

    # Executive Summary
    stats = metrics.get_stats()
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    summary_text = f"""
    <b>Total Images Processed (So Far):</b> {stats['total_images']}/{total_images_in_dataset}<br/>
    <b>Successful:</b> {stats['successful']}<br/>
    <b>Failed:</b> {stats['failed']}<br/>
    <b>Success Rate:</b> {stats['success_rate']}%<br/>
    <b>Total Time:</b> {stats['total_time_seconds']} seconds<br/>
    <b>Avg Time/Image:</b> {stats['avg_time_seconds']} seconds<br/>
    <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    """
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # Metrics Table
    story.append(Paragraph("Performance Metrics", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    table_data = [
        ['Metric', 'Value'],
        ['Images Processed', f"{stats['total_images']}/{total_images_in_dataset}"],
        ['Successful', str(stats['successful'])],
        ['Failed', str(stats['failed'])],
        ['Success Rate (%)', str(stats['success_rate'])],
        ['Total Time (s)', str(stats['total_time_seconds'])],
        ['Avg Time (s)', str(stats['avg_time_seconds'])],
        ['Median Time (s)', str(stats['median_time_seconds'])],
        ['Min Time (s)', str(stats['min_time_seconds'])],
        ['Max Time (s)', str(stats['max_time_seconds'])],
    ]

    table = Table(table_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    # Routes Table
    if stats['routes']:
        story.append(Paragraph("Route Distribution", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))

        route_data = [['Route', 'Count', '%']]
        total = sum(stats['routes'].values())
        for route, count in sorted(stats['routes'].items(), key=lambda x: x[1], reverse=True):
            pct = round(count / total * 100, 1) if total > 0 else 0
            route_data.append([route, str(count), f"{pct}%"])

        route_table = Table(route_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        route_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(route_table)
        story.append(Spacer(1, 0.3*inch))

    # Errors Table
    if stats['errors']:
        story.append(Paragraph("Error Analysis", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))

        error_data = [['Error Type', 'Count', '%']]
        total_errors = sum(stats['errors'].values())
        for error_type, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True):
            pct = round(count / total_errors * 100, 1) if total_errors > 0 else 0
            error_data.append([error_type, str(count), f"{pct}%"])

        error_table = Table(error_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        error_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(error_table)

    # Build PDF
    doc.build(story)
    logger.info(f"✓ PDF saved: {output_file}")

# ============================================================================
# STEP 6: MAIN FUNCTION WITH BATCH MODE
# ============================================================================
def main():
    """Main evaluation function with batch mode"""

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print(" python evaluate_aria_ocid_BATCH_FIXED.py --batch 50 --resume")
        print(" (Resume and process 50 images per batch)")
        print("")
        print(" python evaluate_aria_ocid_BATCH_FIXED.py --batch 100")
        print(" (Start fresh, process 100 images per batch)")
        print("")
        print(" python evaluate_aria_ocid_BATCH_FIXED.py --batch 50 --pilot")
        print(" (Pilot mode: 5 images total, 50 per batch)")
        sys.exit(1)

    # Parse batch size
    batch_size = 50 # Default
    if '--batch' in sys.argv:
        idx = sys.argv.index('--batch')
        if idx + 1 < len(sys.argv):
            try:
                batch_size = int(sys.argv[idx + 1])
            except ValueError:
                print("Error: batch size must be a number")
                sys.exit(1)

    pilot_mode = '--pilot' in sys.argv
    resume_mode = '--resume' in sys.argv
    dataset_path = 'datasets/OCID-dataset'

    # Create results directory
    Path('results').mkdir(exist_ok=True)

    logger.info("="*80)
    logger.info("STARTING ARIA EVALUATION (BATCH MODE - FIXED)")
    logger.info("="*80)
    logger.info(f"Batch Size: {batch_size} images")
    logger.info(f"Resume Mode: {resume_mode}")

    # Load images
    images = load_images(dataset_path, limit=5 if pilot_mode else None)

    if not images:
        logger.error("No images found!")
        sys.exit(1)

    mode = "PILOT (5 images)" if pilot_mode else f"FULL ({len(images)} images)"
    logger.info(f"Mode: {mode}")

    # Load checkpoint if resuming
    checkpoint_file = 'results/.checkpoint.json'
    processed_set = set()
    metrics = MetricsCollector()

    if resume_mode:
        processed_list, checkpoint_metrics = load_checkpoint(checkpoint_file)
        if processed_list is not None and checkpoint_metrics is not None:
            # Restore metrics from checkpoint - FIXED with safe access
            results_list = checkpoint_metrics.get('results', [])
            for result in results_list:
                metrics.add_result(
                    result['image'],
                    result['status'] == 'success',
                    result['time'],
                    result.get('route'),
                    result.get('error')
                )
            processed_set = set(processed_list)
            logger.info(f"✓ Resumed with {len(processed_set)} already processed")
        else:
            logger.warning("Could not load checkpoint, starting fresh")
            processed_set = set()
    else:
        processed_set = set()

    # Filter out already processed images
    images_to_process = [img for img in images if img.name not in processed_set]
    logger.info(f"Images to process: {len(images_to_process)}/{len(images)}")

    if len(images_to_process) == 0:
        logger.info("All images already processed! Nothing to do.")
        sys.exit(0)

    logger.info(f"Starting batch evaluation...")
    logger.info("")

    # Process in batches
    batch_number = 1
    batch_start_idx = 0

    while batch_start_idx < len(images_to_process):
        batch_end_idx = min(batch_start_idx + batch_size, len(images_to_process))
        batch_images = images_to_process[batch_start_idx:batch_end_idx]

        logger.info("")
        logger.info("="*80)
        logger.info(f"BATCH {batch_number}: Processing {len(batch_images)} images")
        logger.info(f"(Images {batch_start_idx + 1}-{batch_end_idx} of {len(images_to_process)})")
        logger.info("="*80)

        batch_start_time = time.time()

        # Process each image in the batch
        for image_path in batch_images:
            try:
                success, exec_time, route, error = run_pipeline_on_image(str(image_path))
                metrics.add_result(image_path.name, success, exec_time, route, error)
                processed_set.add(image_path.name)

                # Save checkpoint after each image
                checkpoint_data = {
                    'processed': list(processed_set),
                    'metrics': {
                        'results': metrics.results,
                        'total_images': metrics.total_images,
                        'successful': metrics.successful,
                        'failed': metrics.failed,
                    }
                }
                save_checkpoint(checkpoint_file, list(processed_set), checkpoint_data['metrics'])

            except KeyboardInterrupt:
                logger.warning("Interrupted! Progress saved to checkpoint.")
                logger.warning(f"Resume with: python evaluate_aria_ocid_BATCH_FIXED.py --batch {batch_size} --resume")
                sys.exit(0)

        batch_time = time.time() - batch_start_time

        # Update PDF report after each batch
        logger.info("")
        logger.info(f"✓ Batch {batch_number} complete in {batch_time:.2f}s")
        logger.info(f" Total progress: {len(processed_set)}/{len(images)} images")
        logger.info(f" Updating PDF report...")

        pdf_file = 'results/ARIA_Evaluation_Report_Partial.pdf'
        generate_pdf_report(pdf_file, metrics, len(images), batch_number=batch_number, batch_size=batch_size)

        logger.info(f"✓ PDF updated: {pdf_file}")

        # Move to next batch
        batch_start_idx = batch_end_idx
        batch_number += 1

    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)

    # Save final JSON results
    stats = metrics.get_stats()
    json_file = f"results/evaluation_results_{'pilot' if pilot_mode else 'full'}.json"

    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': 'pilot' if pilot_mode else 'full',
            'statistics': stats,
            'detailed_results': metrics.results,
        }, f, indent=2)

    logger.info(f"✓ JSON saved: {json_file}")

    # Generate final PDF report
    pdf_file = 'results/ARIA_Evaluation_Report_Partial.pdf'
    generate_pdf_report(pdf_file, metrics, len(images))

    # Delete checkpoint on success
    if Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Images: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']}%")
    print(f"Total Time: {stats['total_time_seconds']}s")
    print(f"Avg Time/Image: {stats['avg_time_seconds']}s")
    print("")
    print(f" Results: {json_file}")
    print(f" Report: {pdf_file}")
    print(f" Logs: results/evaluation.log")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
