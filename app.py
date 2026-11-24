from fastapi.responses import FileResponse, HTMLResponse
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
import ezdxf
import numpy as np
import pypdfium2 as pdfium
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="PDF to DWG Converter MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(tempfile.gettempdir()) / "pdf2dwg_jobs"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 若本機已安裝 ODA File Converter，可改為實際安裝路徑或設定環境變數 ODA_PATH
ODA_EXE = os.environ.get("ODA_PATH") or shutil.which("ODAFileConverter") or r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe"


def is_vector_pdf(pdf_path: Path) -> bool:
    """簡單判斷 PDF 是否有可解析的文字/內容流，作為自動分流依據"""
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


def render_page_image(pdf_path: Path, page_index: int = 0, dpi: int = 250) -> np.ndarray:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil().convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    finally:
        doc.close()


def vector_pdf_to_dxf(pdf_path: Path, dxf_path: Path) -> None:
    """
    向量 PDF 路線：
    目前 MVP 先將第一頁渲染後做線段擷取並輸出為標準 DXF；
    若需高精度工程線段還原，可再接入 PDF 路徑指令解析或專業 CAD 引擎。
    """
    img = render_page_image(pdf_path, page_index=0, dpi=200)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    doc = ezdxf.new(dxfversion="R2013")
    msp = doc.modelspace()
    msp.add_layer("VECTOR_LINES", color=7)

    height, _ = gray.shape
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 轉換影像座標（左上原點）至 CAD 座標（左下原點）
            msp.add_line(
                (float(x1), float(height - y1)),
                (float(x2), float(height - y2)),
                dxfattribs={"layer": "VECTOR_LINES"},
            )

    doc.saveas(str(dxf_path))


def scan_pdf_to_dxf(pdf_path: Path, dxf_path: Path) -> None:
    """
    掃描／一般文件 PDF 路線：
    去雜訊、二值化、輪廓與直線偵測後輸出 DXF。
    """
    img = render_page_image(pdf_path, page_index=0, dpi=250)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=12)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doc = ezdxf.new(dxfversion="R2013")
    msp = doc.modelspace()
    msp.add_layer("SCAN_LINES", color=3)
    msp.add_layer("SCAN_CONTOURS", color=5)

    height, _ = gray.shape

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            msp.add_line(
                (float(x1), float(height - y1)),
                (float(x2), float(height - y2)),
                dxfattribs={"layer": "SCAN_LINES"},
            )

    for cnt in contours:
        if cv2.contourArea(cnt) < 120:
            continue
        pts = [(float(pt[0][0]), float(height - pt[0][1])) for pt in cnt]
        if len(pts) >= 3:
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "SCAN_CONTOURS"})

    doc.saveas(str(dxf_path))


def dxf_to_dwg(dxf_path: Path, out_dir: Path) -> Optional[Path]:
    """透過 ODA File Converter 將 DXF 轉為 DWG（AutoCAD 2013 格式）"""
    if not os.path.exists(ODA_EXE):
        return None

    in_dir = out_dir / "oda_in"
    dwg_out_dir = out_dir / "oda_out"
    in_dir.mkdir(parents=True, exist_ok=True)
    dwg_out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(dxf_path, in_dir / dxf_path.name)

    # ODA 參數: <輸入資料夾> <輸出資料夾> <輸出版本> <輸出格式> <遞迴> <審核> [篩選]
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


@app.post("/api/convert")
async def convert_pdf(
    file: UploadFile = File(...),
    mode: str = Form("auto"),  # auto | vector | scan
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="請上傳 PDF 檔案")

    job_id = str(uuid.uuid4())[:8]
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "input.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

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
        format_used = "DXF (未安裝 ODA，提供相容 CAD 向量檔)"

    return {
        "job_id": job_id,
        "mode_used": detected_mode,
        "format": format_used,
        "download_url": f"/api/download/{job_id}/{download_name}",
        "filename": download_name,
    }


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
            return FileResponse(path=str(candidate), filename=filename, media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail="找不到可下載的檔案")

@app.get("/", response_class=HTMLResponse)
async def home():
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h3>API is running. Please upload index.html to repository root.</h3>"