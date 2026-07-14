from __future__ import annotations

from datetime import datetime
import html
import json
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from medical.compliance import MEDICAL_DISCLAIMER
from medical.cancer_catalog import supported_cancer_labels, supported_cancer_modalities


def build_artifact_stamp() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"


def _as_file_uri(path_value: str | Path | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        return path.resolve().as_uri()
    except OSError:
        return None


def _write_case_report_files(json_path: Path, md_path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["report_html_path"] = str(json_path.with_suffix(".html"))
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    json_path.with_suffix(".html").write_text(_html_report(payload), encoding="utf-8")


def write_case_report(output_dir: str | Path, payload: dict[str, Any], *, generate_pdf: bool = False) -> tuple[Path, Path, Path | None]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_artifact_stamp()
    json_path = target_dir / f"case_report_{stamp}.json"
    md_path = target_dir / f"case_report_{stamp}.md"
    _write_case_report_files(json_path, md_path, payload)
    pdf_path: Path | None = None
    if generate_pdf:
        pdf_path = export_case_pdf(target_dir, payload)
    return json_path, md_path, pdf_path


def update_case_report_case_id(json_path: str | Path, md_path: str | Path, *, case_id: int) -> None:
    json_file = Path(json_path)
    md_file = Path(md_path)
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    payload["case_id"] = case_id
    _write_case_report_files(json_file, md_file, payload)


def export_case_bundle(case_payload: dict[str, Any], export_dir: str | Path, *, include_files: list[str] | None = None, include_pdf: bool = False) -> Path:
    target_dir = Path(export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    case_id = case_payload.get("case_id", "unknown")
    stamp = build_artifact_stamp()
    bundle_path = target_dir / f"medical_case_{case_id}_{stamp}.zip"
    include_files = include_files or []
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("case_summary.json", json.dumps(case_payload, ensure_ascii=False, indent=2))
        archive.writestr("case_summary.md", _markdown_report(case_payload))
        archive.writestr("case_summary.html", _html_report(case_payload))
        if include_pdf:
            pdf_path = export_case_pdf(target_dir, case_payload)
            archive.write(pdf_path, arcname=pdf_path.name)
        for path_str in include_files:
            path = Path(path_str)
            if path.exists() and path.is_file():
                archive.write(path, arcname=path.name)
    return bundle_path


def export_case_pdf(output_dir: str | Path, payload: dict[str, Any], pdf_path: str | Path | None = None) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if pdf_path is None:
        case_id = payload.get("case_id", "unknown")
        stamp = build_artifact_stamp()
        pdf_path = target_dir / f"case_report_{case_id}_{stamp}.pdf"
    pdf_path = Path(pdf_path)

    html_source = _pdf_report_html(payload)

    weasyprint = _try_import("weasyprint")
    if weasyprint is not None:
        weasyprint.HTML(string=html_source, base_url=str(target_dir)).write_pdf(str(pdf_path))
        return pdf_path

    reportlab_modules = _try_import_reportlab()
    if reportlab_modules is not None:
        _render_pdf_with_reportlab(pdf_path, payload, reportlab_modules)
        return pdf_path

    raise ImportError(
        "PDF export requires either 'weasyprint' or 'reportlab', but neither is installed. "
        "Install one with: pip install weasyprint   OR   pip install reportlab"
    )


def _try_import(name: str) -> Any | None:
    try:
        return __import__(name)
    except ImportError:
        return None


def _try_import_reportlab() -> dict[str, Any] | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
        )
        from reportlab.lib.enums import TA_LEFT
        return {
            "colors": colors,
            "A4": A4,
            "getSampleStyleSheet": getSampleStyleSheet,
            "ParagraphStyle": ParagraphStyle,
            "cm": cm,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Paragraph": Paragraph,
            "Spacer": Spacer,
            "Table": Table,
            "TableStyle": TableStyle,
            "PageBreak": PageBreak,
            "TA_LEFT": TA_LEFT,
        }
    except ImportError:
        return None


def _render_pdf_with_reportlab(pdf_path: Path, payload: dict[str, Any], mods: dict[str, Any]) -> None:
    colors = mods["colors"]
    A4 = mods["A4"]
    getSampleStyleSheet = mods["getSampleStyleSheet"]
    ParagraphStyle = mods["ParagraphStyle"]
    cm = mods["cm"]
    SimpleDocTemplate = mods["SimpleDocTemplate"]
    Paragraph = mods["Paragraph"]
    Spacer = mods["Spacer"]
    Table = mods["Table"]
    TableStyle = mods["TableStyle"]
    PageBreak = mods["PageBreak"]
    TA_LEFT = mods["TA_LEFT"]

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#0f172a"))
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#0f172a"))
    body_style = ParagraphStyle("BodyStyle", parent=styles["BodyText"], alignment=TA_LEFT)
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#7f1d1d"),
        backColor=colors.HexColor("#fef2f2"),
        borderColor=colors.HexColor("#fecaca"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    detections = payload.get("detections", [])
    quality_warnings = payload.get("quality_warnings", [])

    summary_data = [
        ["Case ID", str(payload.get("case_id", "-"))],
        ["Risk level", str(payload.get("risk_level", "-"))],
        ["Suspected malignant", str(payload.get("suspected_malignant", False))],
        ["Model", str(payload.get("model_name", "-"))],
        ["Recommendation", str(payload.get("recommendation", "-"))],
    ]
    summary_table = Table([[Paragraph(f"<b>{k}</b>", body_style), Paragraph(html.escape(v), body_style)] for k, v in summary_data], colWidths=[5 * cm, 11 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eff6ff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    findings_rows = [[Paragraph("<b>Label</b>", body_style), Paragraph("<b>Confidence</b>", body_style), Paragraph("<b>BBox</b>", body_style)]]
    if detections:
        for item in detections:
            findings_rows.append([
                Paragraph(html.escape(str(item.get("label", "-"))), body_style),
                Paragraph(f"{item.get('confidence', 0):.2f}", body_style),
                Paragraph(html.escape(str(item.get("bbox", []))), body_style),
            ])
    else:
        findings_rows.append([Paragraph("Không có vùng nghi ngờ nào được ghi nhận.", body_style), Paragraph("", body_style), Paragraph("", body_style)])
    findings_table = Table(findings_rows, colWidths=[5 * cm, 4 * cm, 7 * cm])
    findings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    warning_items = "".join(f"<br/>&bull; {html.escape(str(w))}" for w in quality_warnings) or "Không có cảnh báo chất lượng ảnh."

    story = [
        Paragraph("Medical Imaging Case Report", title_style),
        Spacer(1, 0.4 * cm),
        summary_table,
        Spacer(1, 0.4 * cm),
        Paragraph("Findings", heading_style),
        findings_table,
        Spacer(1, 0.4 * cm),
        Paragraph("Image Quality", heading_style),
        Paragraph(warning_items, body_style),
        PageBreak(),
        Paragraph("Legal Notice", heading_style),
        Paragraph(html.escape(str(payload.get("disclaimer", MEDICAL_DISCLAIMER))), disclaimer_style),
    ]

    doc.build(story)


def _html_report(payload: dict[str, Any]) -> str:
    detections = payload.get("detections", [])
    quality_warnings = payload.get("quality_warnings", [])
    findings_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('label', '-')))}</td><td>{item.get('confidence', 0):.2f}</td><td>{html.escape(str(item.get('bbox', [])))}</td></tr>"
        for item in detections
    ) or '<tr><td colspan="3">Không có vùng nghi ngờ nào được ghi nhận.</td></tr>'
    warning_rows = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in quality_warnings) or "<li>Không có cảnh báo chất lượng ảnh.</li>"
    source_uri = _as_file_uri(payload.get("source_image"))
    processed_uri = _as_file_uri(payload.get("processed_image"))
    source_path = payload.get("source_image") or "-"
    processed_path = payload.get("processed_image") or "-"
    source_section = (
        f'<div class="image-card"><h3>Ảnh gốc</h3><p><strong>Đường dẫn:</strong> {html.escape(str(source_path))}</p><img src="{html.escape(source_uri)}" alt="Source image" /></div>'
        if source_uri
        else f'<div class="image-card"><h3>Ảnh gốc</h3><p><strong>Đường dẫn:</strong> {html.escape(str(source_path))}</p><p>Ảnh gốc không có sẵn để hiển thị.</p></div>'
    )
    processed_section = (
        f'<div class="image-card"><h3>Ảnh đã xử lý / overlay</h3><p><strong>Đường dẫn:</strong> {html.escape(str(processed_path))}</p><img src="{html.escape(processed_uri)}" alt="Processed image" /></div>'
        if processed_uri
        else f'<div class="image-card"><h3>Ảnh đã xử lý / overlay</h3><p><strong>Đường dẫn:</strong> {html.escape(str(processed_path))}</p><p>Ảnh đã xử lý không có sẵn để hiển thị.</p></div>'
    )
    return f"""<!DOCTYPE html>
<html lang=\"vi\">
<head>
  <meta charset=\"utf-8\" />
  <title>Medical Imaging Case Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ color: #0f172a; }}
    .summary {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
    th {{ background: #eff6ff; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 16px; }}
    .image-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; height: auto; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Medical Imaging Case Report</h1>
  <div class=\"summary\">
    <p><strong>Case ID:</strong> {html.escape(str(payload.get('case_id', '-')))}</p>
    <p><strong>Risk level:</strong> {html.escape(str(payload.get('risk_level', '-')))}</p>
    <p><strong>Suspected malignant:</strong> {html.escape(str(payload.get('suspected_malignant', False)))}</p>
    <p><strong>Model:</strong> {html.escape(str(payload.get('model_name', '-')))}</p>
    <p><strong>Recommendation:</strong> {html.escape(str(payload.get('recommendation', '-')))}</p>
  </div>
  <h2>Findings</h2>
  <table>
    <thead><tr><th>Label</th><th>Confidence</th><th>BBox</th></tr></thead>
    <tbody>{findings_rows}</tbody>
  </table>
  <h2>Image Quality</h2>
  <ul>{warning_rows}</ul>
  <div class=\"image-grid\">{source_section}{processed_section}</div>
  <h2>Legal Notice</h2>
  <p>{html.escape(str(payload.get('disclaimer', '')))}</p>
</body>
</html>
"""


def _pdf_report_html(payload: dict[str, Any]) -> str:
    detections = payload.get("detections", [])
    quality_warnings = payload.get("quality_warnings", [])
    findings_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('label', '-')))}</td><td>{item.get('confidence', 0):.2f}</td><td>{html.escape(str(item.get('bbox', [])))}</td></tr>"
        for item in detections
    ) or '<tr><td colspan="3">Không có vùng nghi ngờ nào được ghi nhận.</td></tr>'
    warning_rows = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in quality_warnings) or "<li>Không có cảnh báo chất lượng ảnh.</li>"
    source_uri = _as_file_uri(payload.get("source_image"))
    processed_uri = _as_file_uri(payload.get("processed_image"))
    source_path = payload.get("source_image") or "-"
    processed_path = payload.get("processed_image") or "-"
    source_section = (
        f'<div class="image-card"><h3>Ảnh gốc</h3><p><strong>Đường dẫn:</strong> {html.escape(str(source_path))}</p><img src="{html.escape(source_uri)}" alt="Source image" /></div>'
        if source_uri
        else f'<div class="image-card"><h3>Ảnh gốc</h3><p><strong>Đường dẫn:</strong> {html.escape(str(source_path))}</p><p>Ảnh gốc không có sẵn để hiển thị.</p></div>'
    )
    processed_section = (
        f'<div class="image-card"><h3>Ảnh đã xử lý / overlay</h3><p><strong>Đường dẫn:</strong> {html.escape(str(processed_path))}</p><img src="{html.escape(processed_uri)}" alt="Processed image" /></div>'
        if processed_uri
        else f'<div class="image-card"><h3>Ảnh đã xử lý / overlay</h3><p><strong>Đường dẫn:</strong> {html.escape(str(processed_path))}</p><p>Ảnh đã xử lý không có sẵn để hiển thị.</p></div>'
    )
    disclaimer_text = html.escape(str(payload.get("disclaimer", MEDICAL_DISCLAIMER)))
    return f"""<!DOCTYPE html>
<html lang=\"vi\">
<head>
  <meta charset=\"utf-8\" />
  <title>Medical Imaging Case Report</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, Helvetica, sans-serif; color: #1f2937; font-size: 11pt; line-height: 1.4; margin: 0; padding: 0; }}
    h1 {{ color: #0f172a; font-size: 18pt; margin: 0 0 12pt 0; page-break-after: avoid; }}
    h2 {{ color: #0f172a; font-size: 13pt; margin: 16pt 0 6pt 0; page-break-after: avoid; }}
    .summary {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 12pt; border-radius: 6pt; page-break-inside: avoid; }}
    .summary p {{ margin: 2pt 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 6pt; page-break-inside: auto; }}
    tr {{ page-break-inside: avoid; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 6pt; text-align: left; vertical-align: top; }}
    th {{ background: #eff6ff; }}
    ul {{ margin: 4pt 0; padding-left: 18pt; }}
    .image-grid {{ display: block; margin-top: 10pt; }}
    .image-card {{ border: 1px solid #e2e8f0; border-radius: 6pt; padding: 10pt; margin-bottom: 10pt; page-break-inside: avoid; }}
    .image-card h3 {{ margin: 0 0 4pt 0; font-size: 11pt; }}
    img {{ width: 100%; height: auto; border-radius: 4pt; }}
    .disclaimer {{ color: #7f1d1d; background: #fef2f2; border: 1px solid #fecaca; border-radius: 6pt; padding: 12pt; margin-top: 12pt; page-break-inside: avoid; font-size: 10pt; }}
    .disclaimer h2 {{ color: #7f1d1d; }}
    .page-break {{ page-break-before: always; }}
    @media print {{
      .summary, .image-card, .disclaimer, table {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <h1>Medical Imaging Case Report</h1>
  <div class=\"summary\">
    <p><strong>Case ID:</strong> {html.escape(str(payload.get('case_id', '-')))}</p>
    <p><strong>Risk level:</strong> {html.escape(str(payload.get('risk_level', '-')))}</p>
    <p><strong>Suspected malignant:</strong> {html.escape(str(payload.get('suspected_malignant', False)))}</p>
    <p><strong>Model:</strong> {html.escape(str(payload.get('model_name', '-')))}</p>
    <p><strong>Recommendation:</strong> {html.escape(str(payload.get('recommendation', '-')))}</p>
  </div>
  <h2>Findings</h2>
  <table>
    <thead><tr><th>Label</th><th>Confidence</th><th>BBox</th></tr></thead>
    <tbody>{findings_rows}</tbody>
  </table>
  <h2>Image Quality</h2>
  <ul>{warning_rows}</ul>
  <div class=\"image-grid\">{source_section}{processed_section}</div>
  <div class=\"page-break\"></div>
  <div class=\"disclaimer\">
    <h2>Legal Notice</h2>
    <p>{disclaimer_text}</p>
  </div>
</body>
</html>
"""


def _markdown_report(payload: dict[str, Any]) -> str:
    detections = payload.get("detections", [])
    quality_warnings = payload.get("quality_warnings", [])
    detection_lines = "\n".join(
        f"- {item['label']} | conf={item['confidence']:.2f} | bbox={item['bbox']}" for item in detections
    ) or "- Không có vùng nghi ngờ nào được ghi nhận."
    quality_lines = "\n".join(f"- {warning}" for warning in quality_warnings) or "- Không có cảnh báo chất lượng ảnh."
    supported_targets = ", ".join(supported_cancer_labels())
    supported_modalities = ", ".join(supported_cancer_modalities())
    risk_display = payload.get("risk_level", "-")
    if risk_display == "uncertain":
        risk_display = "uncertain - KET QUA CHUA DU TIN TUONG"
    return (
        "# Medical Imaging Case Report\n\n"
        f"- Case ID: {payload.get('case_id', '-')}\n"
        f"- Risk level: {risk_display}\n"
        f"- Suspected malignant: {payload.get('suspected_malignant', False)}\n"
        f"- Model: {payload.get('model_name', '-')}\n"
        f"- Supported screening targets: {supported_targets}\n"
        f"- Supported modalities: {supported_modalities}\n"
        f"- Source image: {payload.get('source_image', '-')}\n"
        f"- Normalized image: {payload.get('normalized_image', '-')}\n"
        f"- Processed image: {payload.get('processed_image', '-')}\n\n"
        "## Findings\n"
        f"{detection_lines}\n\n"
        "## Image Quality\n"
        f"{quality_lines}\n\n"
        "## Recommendation\n"
        f"{payload.get('recommendation', '-')}\n\n"
        "## Legal Notice\n"
        f"{MEDICAL_DISCLAIMER}\n"
    )
