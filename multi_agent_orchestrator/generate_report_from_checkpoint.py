#!/usr/bin/env python3
"""
Generate Report from Log
Parses results/evaluation.log and generates a PDF report for the evaluation done so far
"""
import re
import sys
from pathlib import Path
from datetime import datetime
import statistics

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

# Log line patterns emitted by evaluate_aria_ocid_CLEAN.py
LOG_PREFIX_RE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (.*)$')
START_RUN_RE = re.compile(r'^STARTING ARIA EVALUATION$')
FOUND_IMAGES_RE = re.compile(r'^Found (\d+) images$')
ROUTE_CHOSEN_RE = re.compile(r'^\[Orchestrator\] Planner chose route: (\S+)$')
SUCCESS_RE = re.compile(r'^✓ (?P<image>\S+) \((?P<route>[^)]*)\) - (?P<time>[\d.]+)s$')
FAILURE_RE = re.compile(r'^✗ (?P<image>\S+) - (?P<error>.+)$')
EVAL_COMPLETE_RE = re.compile(r'^EVALUATION COMPLETE$')

def parse_log(log_file):
    """Parse the evaluation log and return (results, total_expected, is_complete)"""
    if not Path(log_file).exists():
        print(f" Log file not found: {log_file}")
        sys.exit(1)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Only consider the most recent run (log may contain multiple runs)
    last_start = 0
    for i, line in enumerate(lines):
        match = LOG_PREFIX_RE.match(line.rstrip('\n'))
        message = match.group(1) if match else line.rstrip('\n')
        if START_RUN_RE.match(message):
            last_start = i
    lines = lines[last_start:]

    results = []
    total_expected = None
    is_complete = False
    current_route = None

    for line in lines:
        match = LOG_PREFIX_RE.match(line.rstrip('\n'))
        message = match.group(1) if match else line.rstrip('\n')

        found_match = FOUND_IMAGES_RE.match(message)
        if found_match:
            total_expected = int(found_match.group(1))
            continue

        route_match = ROUTE_CHOSEN_RE.match(message)
        if route_match:
            current_route = route_match.group(1)
            continue

        success_match = SUCCESS_RE.match(message)
        if success_match:
            route = success_match.group('route')
            if route == 'unknown' and current_route:
                route = current_route
            results.append({
                'image': success_match.group('image'),
                'status': 'success',
                'time': float(success_match.group('time')),
                'route': route,
                'error': None,
            })
            current_route = None
            continue

        failure_match = FAILURE_RE.match(message)
        if failure_match:
            results.append({
                'image': failure_match.group('image'),
                'status': 'failed',
                'time': 0.0,
                'route': None,
                'error': failure_match.group('error'),
            })
            current_route = None
            continue

        if EVAL_COMPLETE_RE.match(message):
            is_complete = True

    return results, total_expected, is_complete

def generate_report_from_log(log_file, output_pdf):
    """Generate PDF report from evaluation log data"""
    results, total_expected, is_complete = parse_log(log_file)
    if not results:
        print(" No results found in log")
        sys.exit(1)

    # Calculate statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    times = [r['time'] for r in results if r['status'] == 'success']
    routes = {}
    errors = {}
    for r in results:
        if r['status'] == 'success' and r.get('route'):
            routes[r['route']] = routes.get(r['route'], 0) + 1
        elif r['status'] == 'failed' and r.get('error'):
            error_type = r['error'].split(':')[0]
            errors[error_type] = errors.get(error_type, 0) + 1

    # Generate PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    status_label = "Complete" if is_complete else "Partial"
    progress_suffix = f" of {total_expected}" if total_expected else ""

    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("ARIA Pipeline Evaluation Report", styles['Heading1']))
    story.append(Paragraph(f"OCID-Ref Dataset ({status_label}) | {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    status_text = "Evaluation complete" if is_complete else "Evaluation in progress"
    story.append(Paragraph(f"<b>Status:</b> {status_text} - {total}{progress_suffix} images processed", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    summary_text = f"""
    <b>Total Images Processed:</b> {total}<br/>
    <b>Successful:</b> {successful}<br/>
    <b>Failed:</b> {failed}<br/>
    <b>Success Rate:</b> {success_rate:.2f}%<br/>
    <b>Total Time:</b> {sum(times):.2f} seconds<br/>
    <b>Avg Time/Image:</b> {statistics.mean(times):.3f} seconds<br/>
    <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    """
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # Metrics Table
    story.append(Paragraph("Performance Metrics", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    table_data = [
        ['Metric', 'Value'],
        ['Total Images Processed', str(total)],
        ['Successful', str(successful)],
        ['Failed', str(failed)],
        ['Success Rate (%)', f"{success_rate:.2f}"],
        ['Total Time (s)', f"{sum(times):.2f}"],
        ['Avg Time (s)', f"{statistics.mean(times):.3f}"],
        ['Median Time (s)', f"{statistics.median(times):.3f}"],
        ['Min Time (s)', f"{min(times):.3f}"],
        ['Max Time (s)', f"{max(times):.3f}"],
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
    if routes:
        story.append(Paragraph("Route Distribution", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))

        route_data = [['Route', 'Count', '%']]
        total_routed = sum(routes.values())
        for route, count in sorted(routes.items(), key=lambda x: x[1], reverse=True):
            pct = round(count / total_routed * 100, 1) if total_routed > 0 else 0
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
    if errors:
        story.append(Paragraph("Error Analysis", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))

        error_data = [['Error Type', 'Count', '%']]
        total_errors = sum(errors.values())
        for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
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
        story.append(Spacer(1, 0.3*inch))

    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated: {output_pdf}")

if __name__ == '__main__':
    log_file = 'results/evaluation.log'
    output_pdf = 'results/ARIA_Evaluation_Report_Partial.pdf'
    print("Generating report from log...")
    generate_report_from_log(log_file, output_pdf)
    print(f"\n✓ Report saved: {output_pdf}")
    print(f"✓ Log: {log_file}")
