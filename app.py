import os
import json
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
from openai import OpenAI

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# ======================
# Config
# ======================
class Config:
    SHEET_ID = os.getenv("SHEET_ID")
    GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "{}")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

app.config.from_object(Config)

# ======================
# Helpers
# ======================
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def split_keywords(pipe_str: str) -> List[str]:
    return [normalize_text(x) for x in str(pipe_str).split("|") if normalize_text(x)]

def safe_float(x, default=999.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

PRICE_PATTERNS = [
    r"\bүнэ\b", r"\bтөлбөр\b", r"\bхөнгөлөлт\b",
    r"\bapply\b", r"\bpayment\b", r"\bearly\b"
]

def is_price_question(msg: str) -> bool:
    m = normalize_text(msg)
    return any(re.search(p, m) for p in PRICE_PATTERNS)

def has_url(msg: str) -> bool:
    return bool(re.search(r"https?://\S+", msg))

# ======================
# Google Sheets Service
# ======================
class GoogleSheetsService:
    FAQ_TAB = "FAQ_BOLOR"
    COURSE_TAB = "COURSE_BOLOR"

    def __init__(self):
        self.sheet_id = app.config["SHEET_ID"]
        self._cache = {}
        self._lock = threading.Lock()
        self.service = None
        self._init_service()

    def _init_service(self):
        try:
            creds = json.loads(app.config["GOOGLE_CREDENTIALS_JSON"])
            credentials = service_account.Credentials.from_service_account_info(
                creds,
                scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
            )
            self.service = build("sheets", "v4", credentials=credentials)
            logger.info("✅ Google Sheets connected")
        except Exception as e:
            logger.error("❌ Sheets init failed", exc_info=True)

    def _cache_get(self, key):
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            exp, data = item
            return data if datetime.utcnow() < exp else None

    def _cache_set(self, key, data):
        with self._lock:
            self._cache[key] = (
                datetime.utcnow() + timedelta(seconds=app.config["CACHE_TTL"]),
                data
            )

    def _rows_to_dicts(self, values):
        if not values:
            return []
        headers = values[0]
        out = []
        for row in values[1:]:
            item = {}
            for i, h in enumerate(headers):
                item[h] = row[i] if i < len(row) else ""
            if str(item.get("is_active", "true")).lower() in ["true", "1", "yes", ""]:
                out.append(item)
        return out

    def refresh(self):
        if not self.service:
            return [], []
        sheet = self.service.spreadsheets()
        res = sheet.values().batchGet(
            spreadsheetId=self.sheet_id,
            ranges=[f"{self.FAQ_TAB}!A:Z", f"{self.COURSE_TAB}!A:Z"]
        ).execute()

        faq = self._rows_to_dicts(res["valueRanges"][0].get("values", []))
        course = self._rows_to_dicts(res["valueRanges"][1].get("values", []))

        self._cache_set("faq", faq)
        self._cache_set("course", course)
        return faq, course

    def get_faqs(self):
        data = self._cache_get("faq")
        return data if data is not None else self.refresh()[0]

    def get_courses(self):
        data = self._cache_get("course")
        return data if data is not None else self.refresh()[1]

    def match_faq(self, msg):
        msg = normalize_text(msg)
        best, score = None, 0
        for f in self.get_faqs():
            for kw in split_keywords(f.get("q_keywords", "")):
                if re.search(rf"\b{re.escape(kw)}\b", msg):
                    if len(kw) > score:
                        best, score = f, len(kw)
        return best

    def match_course(self, msg):
        msg = normalize_text(msg)
        best, score = None, 0
        for c in self.get_courses():
            for kw in split_keywords(c.get("keywords", "")):
                if re.search(rf"\b{re.escape(kw)}\b", msg):
                    if len(kw) > score:
                        best, score = c, len(kw)
        return best

# ======================
# OpenAI FACT-only Service
# ======================
class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=app.config["OPENAI_API_KEY"])
        self.model = app.config["OPENAI_MODEL"]

    def rewrite(self, question, intent, facts, escalate):
        system = """Та бол Way Academy-гийн албан ёсны AI зөвлөх.
ХАТУУ ДҮРЭМ:
- FACTS JSON-с өөр мэдээлэл ашиглахгүй
- Шинэ зүйл зохиохыг хориглоно
- Нэг intent
- 120–180 үг
"""
        user = f"""
INTENT: {intent}
QUESTION: {question}
FACTS:
{json.dumps(facts, ensure_ascii=False)}

FORMAT:
1 өгүүлбэр хариулт
2–4 bullet
CTA
"""

        if escalate:
            user += "\nДараа нь: 91117577, hello@wayconsulting.io"

        res = self.client.chat.completions.create(
            model=self.model,
            temperature=0.35,
            max_tokens=350,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return res.choices[0].message.content.strip()

# ======================
# Instances
# ======================
sheets = GoogleSheetsService()
ai = OpenAIService()

# ======================
# ManyChat Webhook
# ======================
@app.route("/manychat/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()

        if not msg or has_url(msg):
            return jsonify({"reply": "Асуултаа текстээр бичнэ үү."}), 200

        faq = sheets.match_faq(msg)
        course = sheets.match_course(msg)

        if is_price_question(msg) and not course:
            return jsonify({
                "reply": "Ямар хөтөлбөрийн үнэ сонирхож байна вэ?"
            }), 200

        intent, facts = "FALLBACK", {}

        if course:
            intent = "COURSE"
            facts = course
        elif faq:
            intent = "FAQ"
            facts = {"answer": faq.get("answer")}

        reply = ai.rewrite(
            question=msg,
            intent=intent,
            facts=facts,
            escalate=is_price_question(msg)
        )
        return jsonify({"reply": reply}), 200

    except Exception as e:
        logger.error("Webhook crash", exc_info=True)
        return jsonify({
            "reply": "Системийн алдаа гарлаа. 91117577-р холбогдоорой."
        }), 200

# ======================
# Health
# ======================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "openai": bool(app.config["OPENAI_API_KEY"]),
        "sheets": bool(sheets.service)
    })

# ======================
# Run
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
