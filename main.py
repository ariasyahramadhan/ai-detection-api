from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import logging
import io
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Detection API v2", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ───────────────────────────────────────────
_pipeline = None
_keras_model = None
_load_error = None
_loading = False
_load_lock = threading.Lock()


def _load_models_background():
    global _pipeline, _keras_model, _load_error, _loading
    with _load_lock:
        if _pipeline is not None and _keras_model is not None:
            return
        if _loading:
            return
        _loading = True

    try:
        logger.info("Loading DetectorPipeline...")
        from detector_pipeline import DetectorPipeline
        _pipeline = DetectorPipeline(use_gpu=False)
        logger.info("DetectorPipeline loaded.")

        logger.info("Loading Keras model...")
        import tensorflow as tf
        _keras_model = tf.keras.models.load_model("best_model.keras")
        logger.info("✅ Semua model berhasil dimuat!")
        _load_error = None

    except Exception as e:
        _load_error = str(e)
        logger.error(f"❌ Gagal load model: {e}")
    finally:
        _loading = False


# ── Load model di background saat startup (non-blocking) ────────
@app.on_event("startup")
async def startup_event():
    t = threading.Thread(target=_load_models_background, daemon=True)
    t.start()
    logger.info("Model loading started in background thread.")


# ── Helper: ekstrak teks dari file ──────────────────────────────
def extract_text_from_file(content: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    elif "wordprocessingml" in content_type:
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    elif content_type == "text/plain":
        return content.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Format tidak didukung: {content_type}")


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AI Detection API v2 is running",
        "pipeline_ready": _pipeline is not None,
        "model_ready": _keras_model is not None,
        "loading": _loading,
        "load_error": _load_error,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "model_ready": _keras_model is not None,
        "loading": _loading,
        "load_error": _load_error,
    }


@app.post("/api/detect-ai")
async def detect_ai(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    # Cek apakah model sudah siap (jangan block — loading di background)
    if _pipeline is None or _keras_model is None:
        if _loading:
            raise HTTPException(
                status_code=503,
                detail="Model sedang dimuat, tunggu beberapa saat lalu coba lagi."
            )
        if _load_error:
            raise HTTPException(status_code=503, detail=f"Model gagal dimuat: {_load_error}")
        # Belum ada trigger loading sama sekali, jalankan background
        t = threading.Thread(target=_load_models_background, daemon=True)
        t.start()
        raise HTTPException(
            status_code=503,
            detail="Model sedang dimuat, tunggu beberapa saat lalu coba lagi."
        )

    # 1. Ambil teks input
    input_text = ""
    if file:
        allowed = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ]
        if file.content_type not in allowed:
            raise HTTPException(
                status_code=400,
                detail="Format file tidak didukung. Hanya PDF, DOCX, TXT.",
            )
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File melebihi batas 20MB.")
        try:
            input_text = extract_text_from_file(content, file.content_type)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Gagal membaca file: {str(e)}")
    elif text:
        input_text = text
    else:
        raise HTTPException(status_code=400, detail="Masukkan teks atau upload file.")

    if not input_text or len(input_text.strip()) < 10:
        raise HTTPException(
            status_code=400, detail="Teks terlalu pendek, minimal 10 karakter."
        )

    # 2. Proses pipeline + prediksi
    try:
        import numpy as np

        processed = _pipeline.process_text(input_text)
        x_struct = processed["x_struct"]
        x_embed = processed["x_embed"]

        if _pipeline.scaler is not None:
            x_struct = _pipeline.scaler.transform(x_struct)

        prob = float(_keras_model.predict([x_struct, x_embed], verbose=0)[0][0])

        features = processed["features_dict"]
        word_count = len(input_text.split())
        char_count = len(input_text)

        return {
            "is_ai": prob >= 0.5,
            "ai_probability": round(prob * 100, 2),
            "word_count": word_count,
            "char_count": char_count,
            "metrics": {
                "lexical_diversity_ttr": round(float(features.get("unique_word_ratio", 0)), 4),
                "avg_word_length": round(float(features.get("avg_word_length", 0)), 2),
                "avg_sentence_length": round(float(features.get("avg_sentence_length", 0)), 2),
                "punctuation_rate": round(
                    float(
                        features.get("comma_density", 0)
                        + features.get("colon_density", 0)
                        + features.get("semicolon_density", 0)
                    ),
                    2,
                ),
                "transition_word_rate": round(
                    float(features.get("transition_density", 0)) * 100, 2
                ),
            },
        }

    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses teks: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
