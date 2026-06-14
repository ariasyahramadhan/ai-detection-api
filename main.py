from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Detection API v2", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan URL frontend kamu setelah deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ───────────────────────────────────────────
pipeline = None
keras_model = None


@app.on_event("startup")
async def load_model():
    global pipeline, keras_model
    try:
        import tensorflow as tf
        from detector_pipeline import DetectorPipeline

        logger.info("Loading DetectorPipeline...")
        pipeline = DetectorPipeline(use_gpu=False)

        logger.info("Loading Keras model...")
        keras_model = tf.keras.models.load_model("best_model.keras")

        logger.info("✅ Semua model berhasil dimuat!")
    except Exception as e:
        logger.error(f"❌ Gagal load model: {e}")


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
        "pipeline_ready": pipeline is not None,
        "model_ready": keras_model is not None,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "model_ready": keras_model is not None,
    }


@app.post("/api/detect-ai")
async def detect_ai(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if pipeline is None or keras_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum siap, coba beberapa saat lagi.",
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

        processed = pipeline.process_text(input_text)

        x_struct = processed["x_struct"]  # (1, n_features)
        x_embed = processed["x_embed"]   # (1, 384)

        # Apply scaler jika tersedia
        if pipeline.scaler is not None:
            x_struct = pipeline.scaler.transform(x_struct)

        # Prediksi dengan Keras model
        prob = float(keras_model.predict([x_struct, x_embed], verbose=0)[0][0])

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
