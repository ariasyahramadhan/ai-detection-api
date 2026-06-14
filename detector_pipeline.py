import json
import re
import os
import pickle
import numpy as np
from pathlib import Path

# Jangan mengimport sentence_transformers di top level
# untuk menghindari load lambat atau GPU issue saat di-import oleh skrip lain.

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
WORDLIST_DIR        = "wordlists"
FEATURE_CONFIG_FILE = "feature_config.json"
SCALER_FILE         = "structural_scaler.pkl"

SUBSTITUTIONS = {
    r'(?i)\bgeneric_name\b': 'Alex',
    r'(?i)\bgeneric_namehad\b': 'Alex had',
    r'(?i)\bgeneric_city\b': 'London',
    r'(?i)\bgeneric_citynbsp\b': 'London',
    r'(?i)\bgeneric_school\b': 'University'
}

EN_STOPWORDS = {'and', 'the', 'is', 'in', 'of', 'to', 'this', 'that', 'it', 'for'}
ID_STOPWORDS = {'dan', 'yang', 'di', 'dari', 'ke', 'ini', 'itu', 'pada', 'dengan', 'untuk'}

# Default kanonik
ALL_FEATURES = [
    "sentence_count", "avg_sentence_length", "sentence_length_std",
    "paragraph_count", "paragraph_length_std", "unique_word_ratio",
    "em_dash_density", "colon_density", "semicolon_density",
    "exclamation_density", "question_density", "bullet_list_count",
    "bold_pattern_count", "header_pattern_count", "avg_word_length",
    "comma_density", "parenthesis_density", "contraction_count",
    "number_density", "capital_ratio",
    "ai_vocab_density", "hedging_density", "transition_density",
    "formulaic_score", "certainty_opener", "vague_attribution_density",
    "not_only_but_also", "formal_tone_score", "copypaste_artifact_count"
]

# Regex patterns (compiled)
_RE_SENTENCES   = re.compile(r"[.!?]+")
_RE_WORDS       = re.compile(r"\b\w+\b")
_RE_BULLET      = re.compile(r"^\s*([\u2022\-\*]|\d+\.?)\s+", re.MULTILINE)
_RE_BOLD        = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_RE_HEADER      = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_NUMBER      = re.compile(r"\b\d+\b")
_EM_DASH        = "\u2014"


class DetectorPipeline:
    """
    Pipeline lengkap untuk inference deteksi tulisan AI.

    Alur:
        raw_text
            → clean_text()
            → detect_language()
            → extract_features()           → raw 29-dim float array
            → scaler.transform()           → scaled 29-dim float array  (WAJIB!)
            → SentenceTransformer.encode() → 384-dim L2-normalized vector (WAJIB!)
            → (scaled_struct, embed)       → siap untuk model.predict()
    """

    def __init__(self, use_gpu=True):
        self.wordlists       = {}
        self.active_features = ALL_FEATURES.copy()
        self.embedding_model = None
        self.scaler          = None          # StandardScaler — dimuat dari SCALER_FILE
        self.device          = "cuda" if use_gpu else "cpu"

        self._load_config()
        self._load_wordlists()
        self._load_scaler()
        self._load_embedding_model()

    # ----------------------------------------------------------
    # Loaders
    # ----------------------------------------------------------

    def _load_config(self):
        if os.path.exists(FEATURE_CONFIG_FILE):
            try:
                with open(FEATURE_CONFIG_FILE, encoding="utf-8") as f:
                    cfg = json.load(f)
                self.active_features = cfg.get("active_features", ALL_FEATURES)
            except Exception as e:
                print(f"Warning: Gagal membaca {FEATURE_CONFIG_FILE}. Menggunakan ALL_FEATURES. Error: {e}")

    def _load_scaler(self):
        """
        Muat StandardScaler yang di-fit saat training (Notebook 4).
        Scaler WAJIB ada agar fitur struktural memiliki distribusi yang sama
        seperti saat training. Tanpa scaler, prediksi akan selalu ekstrem (0 atau 1).
        """
        if os.path.exists(SCALER_FILE):
            try:
                with open(SCALER_FILE, "rb") as f:
                    self.scaler = pickle.load(f)
                print(f"[Pipeline] StandardScaler dimuat dari {SCALER_FILE} "
                      f"(n_features={self.scaler.n_features_in_})")
            except Exception as e:
                print(f"[Pipeline] Warning: Gagal memuat scaler dari {SCALER_FILE}. "
                      f"Prediksi akan tidak akurat! Error: {e}")
        else:
            print(f"[Pipeline] Warning: {SCALER_FILE} tidak ditemukan. "
                  f"Jalankan Notebook 4 (training) terlebih dahulu. "
                  f"Prediksi TIDAK akan akurat tanpa scaler!")

    def _flat_set(self, data, key):
        raw = data.get(key, [])
        if isinstance(raw, list):
            return set(w.lower() for w in raw)
        return set()

    def _nested_set(self, data, key):
        result = set()
        nested = data.get(key, {})
        if isinstance(nested, dict):
            for cat_words in nested.values():
                if isinstance(cat_words, list):
                    result.update(w.lower() for w in cat_words)
        elif isinstance(nested, list):
            result.update(w.lower() for w in nested)
        return result

    def _load_wordlists(self):
        wl = {}
        p  = Path(WORDLIST_DIR)

        try:
            with open(p / "ai_overrepresented_vocab.json", encoding="utf-8") as f: d = json.load(f)
            wl["ai_vocab_en"] = self._flat_set(d, "en")
            wl["ai_vocab_id"] = self._flat_set(d, "id")

            with open(p / "hedging_phrases.json", encoding="utf-8") as f: d = json.load(f)
            wl["hedging_en"] = self._flat_set(d, "en")
            wl["hedging_id"] = self._flat_set(d, "id")

            with open(p / "transition_phrases.json", encoding="utf-8") as f: d = json.load(f)
            wl["transition_en"] = self._nested_set(d, "en")
            wl["transition_id"] = self._nested_set(d, "id")

            with open(p / "formulaic_phrases.json", encoding="utf-8") as f: d = json.load(f)
            wl["formulaic_en"] = self._nested_set(d, "en")
            wl["formulaic_id"] = self._nested_set(d, "id")

            with open(p / "certainty_openers.json", encoding="utf-8") as f: d = json.load(f)
            wl["certainty_en"] = self._flat_set(d, "en")
            wl["certainty_id"] = self._flat_set(d, "id")

            with open(p / "vague_attribution.json", encoding="utf-8") as f: d = json.load(f)
            wl["vague_en"] = self._flat_set(d, "en")
            wl["vague_id"] = self._flat_set(d, "id")

            with open(p / "formal_register_pairs.json", encoding="utf-8") as f: d = json.load(f)
            wl["formal_pairs_en"] = d.get("en", [])
            wl["formal_pairs_id"] = d.get("id", [])

            with open(p / "negative_parallelism.json", encoding="utf-8") as f: d = json.load(f)
            wl["neg_parallel_en"] = self._flat_set(d, "en")
            wl["neg_parallel_id"] = self._flat_set(d, "id")

            with open(p / "copypaste_artifacts.json", encoding="utf-8") as f: d = json.load(f)
            raw_patterns      = d.get("patterns", d) if isinstance(d, dict) else d
            compiled_patterns = []
            for pat in raw_patterns:
                try:
                    compiled_patterns.append({
                        "name"     : pat.get("name", ""),
                        "severity" : pat.get("severity", "moderate"),
                        "_compiled": re.compile(pat["regex"], re.IGNORECASE | re.MULTILINE)
                    })
                except (re.error, KeyError):
                    pass
            wl["artifact_patterns"] = compiled_patterns

            with open(p / "contractions.json", encoding="utf-8") as f: d = json.load(f)
            contractions_flat = set()
            en_block = d.get("en", d)
            if isinstance(en_block, dict):
                for cat_words in en_block.values():
                    if isinstance(cat_words, list):
                        contractions_flat.update(w.lower() for w in cat_words)
            elif isinstance(en_block, list):
                contractions_flat.update(w.lower() for w in en_block)
            wl["contractions_flat"] = contractions_flat

            self.wordlists = wl
        except Exception as e:
            print(f"Warning: Gagal memuat sebagian wordlists. Error: {e}")

    def _load_embedding_model(self):
        """
        Load SentenceTransformer MiniLM.
        PENTING: normalize_embeddings=True wajib digunakan saat encode()
        agar konsisten dengan training di Notebook 3.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L6-v2',
                device=self.device
            )
            # Verifikasi: encode satu teks dummy dan cek normanya
            test_vec  = self.embedding_model.encode(
                ["test"], convert_to_numpy=True, normalize_embeddings=True
            )
            test_norm = float(np.linalg.norm(test_vec[0]))
            print(f"[Pipeline] MiniLM dimuat. L2 norm test: {test_norm:.4f} (harus ≈ 1.0)")
        except Exception as e:
            print(f"[Pipeline] Warning: Gagal memuat SentenceTransformer. Error: {e}")

    # ----------------------------------------------------------
    # Text Processing
    # ----------------------------------------------------------

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.replace('\r', '').strip()
        for pattern, replacement in SUBSTITUTIONS.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r' +', ' ', text)
        return text

    def detect_language(self, text):
        text_lower = text.lower()
        words      = set(re.findall(r'\b\w+\b', text_lower))
        id_score   = len(words.intersection(ID_STOPWORDS))
        en_score   = len(words.intersection(EN_STOPWORDS))
        if id_score > en_score and id_score > 0:
            return 'id'
        return 'en'

    def count_phrases(self, text_lower, phrase_set):
        count = 0
        for phrase in phrase_set:
            phrase = phrase.lower()
            if " " in phrase:
                count += text_lower.count(phrase)
            else:
                count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
        return count

    def compute_formal_score(self, text_lower, formal_pairs):
        formal_total   = 0
        informal_total = 0
        for pair in formal_pairs:
            f = pair.get("formal",   "").lower()
            i = pair.get("informal", "").lower()
            if f:
                if " " in f: formal_total   += text_lower.count(f)
                else:        formal_total   += len(re.findall(r"\b" + re.escape(f) + r"\b", text_lower))
            if i:
                if " " in i: informal_total += text_lower.count(i)
                else:        informal_total += len(re.findall(r"\b" + re.escape(i) + r"\b", text_lower))
        total = formal_total + informal_total
        return formal_total / total if total > 0 else 0.0

    # ----------------------------------------------------------
    # Feature Extraction
    # ----------------------------------------------------------

    def extract_features(self, text, lang="en"):
        """
        Ekstrak 29 fitur raw dari teks (BELUM di-scale).
        Scaler akan diterapkan terpisah di process_text().

        Returns:
            features_dict   : dict {feature_name: raw_float_value}
            struct_array_raw: np.ndarray shape (1, n_active_features), dtype float32, BELUM di-scale
        """
        if not isinstance(text, str) or not text.strip():
            features = {feat: 0.0 for feat in ALL_FEATURES}
            return features, np.zeros((1, len(self.active_features)), dtype=np.float32)

        text_lower = text.lower()
        char_count = max(len(text), 1)

        sentences  = [s.strip() for s in _RE_SENTENCES.split(text) if s.strip()]
        words      = _RE_WORDS.findall(text_lower)
        word_count = max(len(words), 1)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        features = {}

        features["sentence_count"]      = float(len(sentences))
        sent_lens = [len(_RE_WORDS.findall(s)) for s in sentences]
        features["avg_sentence_length"] = float(np.mean(sent_lens)) if sent_lens else 0.0
        features["sentence_length_std"] = float(np.std(sent_lens))  if len(sent_lens) > 1 else 0.0

        features["paragraph_count"]      = float(len(paragraphs))
        para_lens = [len(_RE_WORDS.findall(p)) for p in paragraphs]
        features["paragraph_length_std"] = float(np.std(para_lens)) if len(para_lens) > 1 else 0.0

        features["unique_word_ratio"]    = len(set(words)) / word_count
        features["em_dash_density"]      = text.count(_EM_DASH) / char_count * 1000
        features["colon_density"]        = text.count(":") / char_count * 1000
        features["semicolon_density"]    = text.count(";") / char_count * 1000
        features["exclamation_density"]  = text.count("!") / char_count * 1000
        features["question_density"]     = text.count("?") / char_count * 1000

        features["bullet_list_count"]    = float(len(_RE_BULLET.findall(text)))
        features["bold_pattern_count"]   = float(len(_RE_BOLD.findall(text)))
        features["header_pattern_count"] = float(len(_RE_HEADER.findall(text)))

        features["avg_word_length"]      = float(np.mean([len(w) for w in words])) if words else 0.0
        features["comma_density"]        = text.count(",") / char_count * 1000
        features["parenthesis_density"]  = (text.count("(") + text.count(")")) / char_count * 1000

        wl = self.wordlists
        if lang == "en" and wl:
            features["contraction_count"] = float(
                sum(1 for w in words if w in wl.get("contractions_flat", set()))
            )
        else:
            features["contraction_count"] = 0.0

        features["number_density"] = len(_RE_NUMBER.findall(text)) / char_count * 1000

        alpha_chars = [c for c in text if c.isalpha()]
        features["capital_ratio"] = (
            sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        )

        lang_key = lang if lang in ("en", "id") else "en"

        if wl:
            ai_vocab = wl.get(f"ai_vocab_{lang_key}", set())
            features["ai_vocab_density"] = sum(1 for w in words if w in ai_vocab) / word_count

            features["hedging_density"]           = self.count_phrases(text_lower, wl.get(f"hedging_{lang_key}",    set())) / word_count
            features["transition_density"]         = self.count_phrases(text_lower, wl.get(f"transition_{lang_key}", set())) / word_count
            features["formulaic_score"]            = self.count_phrases(text_lower, wl.get(f"formulaic_{lang_key}",  set())) / word_count
            features["certainty_opener"]           = float(self.count_phrases(text_lower[:200], wl.get(f"certainty_{lang_key}", set())))
            features["vague_attribution_density"]  = self.count_phrases(text_lower, wl.get(f"vague_{lang_key}",     set())) / word_count
            features["not_only_but_also"]          = float(self.count_phrases(text_lower, wl.get(f"neg_parallel_{lang_key}", set())))
            features["formal_tone_score"]          = self.compute_formal_score(text_lower, wl.get(f"formal_pairs_{lang_key}", []))

            artifact_count = 0
            for pat in wl.get("artifact_patterns", []):
                compiled = pat.get("_compiled")
                if compiled:
                    artifact_count += len(compiled.findall(text))
            features["copypaste_artifact_count"] = float(artifact_count)
        else:
            for feat in ALL_FEATURES[20:]:
                features[feat] = 0.0

        # Susun array sesuai urutan active_features
        struct_array_raw = np.array(
            [features.get(f, 0.0) for f in self.active_features],
            dtype=np.float32
        ).reshape(1, -1)  # shape (1, n_features)

        return features, struct_array_raw

    # ----------------------------------------------------------
    # Main Entry Point
    # ----------------------------------------------------------

    def process_text(self, raw_text, force_lang=None):
        """
        Proses satu teks raw menjadi input siap untuk model.

        Returns dict:
            language      : 'en' atau 'id'
            text_clean    : teks setelah pembersihan
            features_dict : nilai raw setiap fitur (untuk display)
            x_struct      : np.ndarray (1, n_features), float32, RAW (belum di-scale)
                            → subprocess akan menyimpan ini, dan Notebook 5 yang akan apply scaler
            x_embed       : np.ndarray (1, 384), float32, L2-normalized
        """
        text_clean  = self.clean_text(raw_text)
        lang        = force_lang if force_lang in ('en', 'id') else self.detect_language(text_clean)

        # Ekstrak fitur struktural (raw, belum di-scale)
        features_dict, struct_array_raw = self.extract_features(text_clean, lang=lang)

        # Encode embedding (L2-normalized — HARUS konsisten dengan Notebook 3)
        if self.embedding_model:
            embed_array = self.embedding_model.encode(
                [text_clean],
                convert_to_numpy=True,
                normalize_embeddings=True  # WAJIB: konsisten dengan training
            )
        else:
            embed_array = np.zeros((1, 384), dtype=np.float32)

        return {
            "language"     : lang,
            "text_clean"   : text_clean,
            "features_dict": features_dict,
            "x_struct"     : struct_array_raw,   # (1, n_features) — RAW, scaler applied di NB5
            "x_embed"      : embed_array          # (1, 384) — L2-normalized
        }