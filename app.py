import os
import shutil
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Optional

import cv2
import ezdxf
import numpy as np
import pypdfium2 as pdfium
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI(title="PDF to DWG Converter MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(tempfile.gettempdir()) / "pdf2dwg_jobs"
BASE_DIR.mkdir(parents=True, exist_ok=True)

ODA_EXE = (
    os.environ.get("ODA_PATH")
    or shutil.which("ODAFileConverter")
    or r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe"
)


def ensure_layer(doc: ezdxf.document.Drawing, name: str, color: int = 7) -> None:
    if name not in doc.layers:
        doc.layers.new(name=name, dxfattribs={"color": color})


def is_vector_pdf(pdf_path: Path) -> bool:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(min(len(doc), 3)):
            page = doc[i]
            textpage = page.get_textpage()
            txt = textpage.get_text_range().strip()
            if len(txt) > 10:
                return True
        return False
    finally:
        doc.close()


def render_page_image(pdf_path: Path, page_index: int = 0, dpi: int = 150) -> np.ndarray:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(doc) == 0:
            raise ValueError("PDF 沒有任何頁面")
        safe_index = min(max(page_index, 0), len(doc) - 1)
        page = doc[safe_index]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil().convert("RGB")
        arr = np.array(pil_img)

        max_dim = 2400
        h, w, _ = arr.shape
        if max(h, w) > max_dim:
            ratio = max_dim / float(max(h, w))
            arr = cv2.resize(arr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        doc.close()


def add_lines_to_modelspace(
    msp,
    lines,
    layer_name: str,
    height: int,
    max_lines: int = 3000,
) -> None:
    if lines is None:
        return
    for line in lines[:max_lines]:
        pts = np.asarray(line).reshape(-1)
        if pts.size >= 4:
            x1, y1, x2, y2 = float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])
            msp.add_line(
                (x1, float(height) - y1),
                (x2, float(height) - y2),
                dxfattribs={"layer": layer_name},
            )


def vector_pdf_to_dxf(pdf_path: Path, dxf_path: Path) -> None:
    img = render_page_image(pdf_path, page_index=0, dpi=150)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    doc = ezdxf.new(dxfversion="R2013")
    msp = doc.modelspace()
    ensure_layer(doc, "VECTOR_LINES", color=7)

    height, _ = gray.shape
    add_lines_to_modelspace(msp, lines, "VECTOR_LINES", height, max_lines=3000)

    doc.saveas(str(dxf_path))


def scan_pdf_to_dxf(pdf_path: Path, dxf_path: Path) -> None:
    img = render_page_image(pdf_path, page_index=0, dpi=150)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=12)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doc = ezdxf.new(dxfversion="R2013")
    msp = doc.modelspace()
    ensure_layer(doc, "SCAN_LINES", color=3)
    ensure_layer(doc, "SCAN_CONTOURS", color=5)

    height, _ = gray.shape
    add_lines_to_modelspace(msp, lines, "SCAN_LINES", height, max_lines=3000)

    added_contours = 0
    for cnt in contours:
        if added_contours >= 1500:
            break
        if cv2.contourArea(cnt) < 150:
            continue
        pts = [(float(pt[0][0]), float(height) - float(pt[0][1])) for pt in cnt]
        if len(pts) >= 3:
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "SCAN_CONTOURS"})
            added_contours += 1

    doc.saveas(str(dxf_path))


def dxf_to_dwg(dxf_path: Path, out_dir: Path) -> Optional[Path]:
    if not os.path.exists(ODA_EXE):
        return None

    in_dir = out_dir / "oda_in"
    dwg_out_dir = out_dir / "oda_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    dwg_out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(dxf_path, in_dir / dxf_path.name)

    cmd = [
        ODA_EXE,
        str(in_dir),
        str(dwg_out_dir),
        "ACAD2013",
        "DWG",
        "0",
        "1",
        "*.DXF",
    ]
    subprocess.run(cmd, check=True, timeout=120)

    dwg_file = dwg_out_dir / (dxf_path.stem + ".dwg")
    return dwg_file if dwg_file.exists() else None


@app.get("/", response_class=HTMLResponse)
async def home():
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h3>API is running. Please upload index.html to repository root.</h3>"


@app.post("/api/convert")
async def convert_pdf(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="請上傳 PDF 檔案")

    job_id = str(uuid.uuid4())[:8]
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="檔案超過 25MB 上限，請改用較小的 PDF 測試")

        pdf_path = job_dir / "input.pdf"
        with open(pdf_path, "wb") as f:
            f.write(content)

        if mode == "auto":
            detected_mode = "vector" if is_vector_pdf(pdf_path) else "scan"
        else:
            detected_mode = mode

        dxf_path = job_dir / "output.dxf"
        if detected_mode == "vector":
            vector_pdf_to_dxf(pdf_path, dxf_path)
        else:
            scan_pdf_to_dxf(pdf_path, dxf_path)

        dwg_path = dxf_to_dwg(dxf_path, job_dir)
        if dwg_path:
            final_file = dwg_path
            download_name = Path(file.filename).stem + ".dwg"
            format_used = "DWG (via ODA)"
        else:
            final_file = dxf_path
            download_name = Path(file.filename).stem + ".dxf"
            format_used = "DXF (相容 CAD 向量檔)"

        if not final_file.exists():
            raise RuntimeError("輸出檔案未成功產生")

        return {
            "job_id": job_id,
            "mode_used": detected_mode,
            "format": format_used,
            "download_url": f"/api/download/{job_id}/{download_name}",
            "filename": download_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"轉檔處理失敗：{str(exc)}")


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    job_dir = BASE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="找不到轉檔任務")

    for candidate in [
        job_dir / "oda_out" / (Path(filename).stem + ".dwg"),
        job_dir / "output.dxf",
    ]:
        if candidate.exists():
            return FileResponse(
                path=str(candidate),
                filename=filename,
                media_type="application/octet-stream",
            )

    raise HTTPException(status_code=404, detail="找不到可下載的檔案")