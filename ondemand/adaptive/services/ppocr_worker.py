import json
import os
import sys
import tempfile
import time

DPI, MAX_SIDE = 200, 4000
CHANNEL = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)


def emit(payload):
    CHANNEL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    CHANNEL.flush()


def render(path, index, target):
    import fitz
    with fitz.open(path) as doc:
        page = doc[index]
        dpi = DPI
        side = max(page.rect.width, page.rect.height) * dpi / 72
        if side > MAX_SIDE:
            dpi = int(dpi * MAX_SIDE / side)
        page.get_pixmap(dpi=dpi).save(target)
    return dpi


def main():
    started = time.perf_counter()
    try:
        from paddleocr import PaddleOCR
        device = os.environ.get("PPOCR_DEVICE", "gpu:0")
        ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det",
                        text_recognition_model_name="PP-OCRv5_server_rec",
                        use_doc_orientation_classify=True, use_doc_unwarping=False,
                        use_textline_orientation=False, device=device)
    except Exception as error:
        emit({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    emit({"ready": True, "device": device, "load_seconds": round(time.perf_counter() - started, 3)})
    folder = tempfile.mkdtemp(prefix="ppocr_")
    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        started = time.perf_counter()
        target = os.path.join(folder, "page.png")
        try:
            dpi = render(job["path"], int(job["index"]), target)
            rendered = time.perf_counter()
            text, lines = [], []
            for result in ocr.predict(target):
                text.extend(result["rec_texts"])
                polys = result.get("rec_polys", result.get("dt_polys", []))
                for value, score, poly in zip(result["rec_texts"], result["rec_scores"], polys):
                    lines.append({"text": value, "score": float(score),
                                  "poly": [[float(x), float(y)] for x, y in poly]})
            emit({"page_id": job["page_id"], "text": "\n".join(text), "lines": lines, "dpi": dpi,
                  "render_seconds": round(rendered - started, 6),
                  "seconds": round(time.perf_counter() - started, 6), "error": None})
        except Exception as error:
            emit({"page_id": job["page_id"], "text": "", "lines": [], "seconds": round(time.perf_counter() - started, 6),
                  "error": f"{type(error).__name__}: {error}"[:300]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
