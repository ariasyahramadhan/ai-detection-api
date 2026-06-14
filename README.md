# AI Detection API v2

Backend untuk deteksi teks AI menggunakan Keras deep learning + SentenceTransformer.

## Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Status server |
| GET | `/api/health` | Cek kesiapan model |
| POST | `/api/detect-ai` | Deteksi teks AI |

## POST `/api/detect-ai`

**Form Data:**
- `text` (string, opsional) — teks langsung
- `file` (file, opsional) — file PDF / DOCX / TXT

**Response:**
```json
{
  "is_ai": true,
  "ai_probability": 87.5,
  "word_count": 320,
  "char_count": 1850,
  "metrics": {
    "lexical_diversity_ttr": 0.42,
    "avg_word_length": 5.3,
    "avg_sentence_length": 18.2,
    "punctuation_rate": 12.5,
    "transition_word_rate": 3.1
  }
}
```

## File yang Dibutuhkan

Letakkan file berikut di root folder sebelum deploy:
- `best_model.keras` — model Keras hasil training
- `structural_scaler.pkl` — StandardScaler dari training (jika ada)
- `feature_config.json` — konfigurasi fitur aktif (jika ada)
- `wordlists/` — folder wordlist untuk DetectorPipeline (jika ada)

## Deploy ke Railway

1. Push repo ini ke GitHub
2. Buat project baru di Railway → "Deploy from GitHub repo"
3. Railway akan otomatis build via Dockerfile
4. Generate domain di Settings → Networking
5. Tambahkan env var `VITE_AI_DETECTION_API_URL` di frontend dengan URL domain tersebut
