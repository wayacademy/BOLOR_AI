import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from google.oauth2 import service_account
from googleapiclient.discovery import build

from cachetools import TTLCache

from openai import OpenAI


# ======================
# Logging
# ======================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("way-bot")


# ======================
# Config
# ======================
class Config:
    # Server
    PORT = int(os.getenv("PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # Google Sheets
    SHEET_ID = os.getenv("SHEET_ID")
    GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

    # Cache
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Safety/time budget (ManyChat external request ~10s)
    # We keep our own budget slightly lower.
    TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "8.5"))

    # Response limits
    MAX_COURSES_IN_CONTEXT = int(os.getenv("MAX_COURSES_IN_CONTEXT", "3"))
    MAX_FAQS_IN_CONTEXT = int(os.getenv("MAX_FAQS_IN_CONTEXT", "5"))
    MAX_DESC_CHARS = int(os.getenv("MAX_DESC_CHARS", "260"))


# ======================
# App init
# ======================
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)


# ======================
# Helpers
# ======================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def manychat_text(text: str):
    """Dynamic Block (v2) response."""
    return jsonify({
        "version": "v2",
        "content": {
            "messages": [{"type": "text", "text": text}]
        }
    })


def clamp(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n].rstrip() + "..."


# ======================
# Google Sheets Service (TTL Cache)
# ======================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


class GoogleSheetsService:
    def __init__(self, sheet_id: str, credentials_json_str: str, cache_ttl: int = 300):
        self.sheet_id = sheet_id
        self.cache = TTLCache(maxsize=16, ttl=cache_ttl)
        self.service = self._init_service(credentials_json_str)

    def _init_service(self, credentials_json_str: str):
        info = json.loads(credentials_json_str)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
        logger.info("✅ Google Sheets API initialized")
        return svc

    def _read_values(self, sheet_name: str, a1_range: str = "A:Z") -> List[List[str]]:
        resp = (
            self.service.spreadsheets()
            .values()
            .get(spreadsheetId=self.sheet_id, range=f"{sheet_name}!{a1_range}")
            .execute()
        )
        return resp.get("values", [])

    def get_sheet_dicts(self, sheet_name: str) -> List[Dict[str, Any]]:
        cache_key = f"sheet:{sheet_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        values = self._read_values(sheet_name)
        if not values:
            self.cache[cache_key] = []
            return []

        headers = values[0]
        out: List[Dict[str, Any]] = []

        for row in values[1:]:
            item = {h: (row[i] if i < len(row) else "") for i, h in enumerate(headers)}

            # Filter is_active if present
            is_active = str(item.get("is_active", "True")).strip().lower() == "true"
            if not is_active:
                continue

            out.append(item)

        self.cache[cache_key] = out
        logger.info(f"✅ Loaded {len(out)} rows from '{sheet_name}' (cached)")
        return out

    @staticmethod
    def _safe_float(x: Any, default: float = 999.0) -> float:
        try:
            return float(str(x).strip())
        except Exception:
            return default

    def get_all_faqs(self) -> List[Dict[str, Any]]:
        faqs = self.get_sheet_dicts("faq")
        return sorted(faqs, key=lambda r: self._safe_float(r.get("priority", 999)))

    def get_all_courses(self) -> List[Dict[str, Any]]:
        courses = self.get_sheet_dicts("courses")
        return sorted(courses, key=lambda r: self._safe_float(r.get("priority", 999)))

    def get_course_by_keyword(self, user_text: str) -> Optional[Dict[str, Any]]:
        t = (user_text or "").lower().strip()
        if not t:
            return None

        for c in self.get_all_courses():
            kw = (c.get("keywords") or "").lower()
            if kw:
                kws = [k.strip() for k in kw.split("|") if k.strip()]
                if any(k in t for k in kws):
                    return c

            name = (c.get("course_name") or "").lower().strip()
            if name and name in t:
                return c

        return None


# ======================
# AI Service (OpenAI)
# ======================
class AIService:
    def __init__(self, api_key: Optional[str], model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY missing; AI will be disabled")

    def build_system_prompt(self) -> str:
        return (
            "Та бол Way Academy-гийн албан ёсны зөвлөх чатбот.\n"
            "Дүрэм:\n"
            "1) ЗӨВХӨН өгөгдсөн контекстээс хариул.\n"
            "2) Монгол хэлээр, товч, ойлгомжтой бич.\n"
            "3) Үнэ, хугацаа, хуваарь, төлбөрийн нөхцөлийг тодорхой харуул.\n"
            "4) Хэрэв контекстэд байхгүй бол: 'Уучлаарай, энэ асуултад одоогоор тодорхой хариулт олдсонгүй.' гэж хэл.\n"
        )

    def format_context(self, courses: List[Dict[str, Any]], faqs: List[Dict[str, Any]]) -> str:
        parts: List[str] = []

        if courses:
            parts.append("=== COURSES ===")
            for c in courses:
                parts.append(
                    "\n".join([
                        f"course_id: {c.get('course_id','')}",
                        f"course_name: {c.get('course_name','')}",
                        f"teacher: {c.get('teacher','')}",
                        f"duration: {c.get('duration','')}",
                        f"schedule_1: {c.get('schedule_1','')}",
                        f"schedule_2: {c.get('schedule_2','')}",
                        f"price_full: {c.get('price_full','')}",
                        f"price_discount: {c.get('price_discount','')}",
                        f"price_discount_until: {c.get('price_discount_until','')}",
                        f"payment_options: {c.get('payment_options','')}",
                        f"application_link: {c.get('application_link','')}",
                        f"cta_caption: {c.get('cta_caption','')}",
                        f"description: {clamp(c.get('description',''), app.config['MAX_DESC_CHARS'])}",
                        "---"
                    ])
                )

        if faqs:
            parts.append("\n=== FAQ ===")
            for f in faqs:
                parts.append(
                    "\n".join([
                        f"faq_id: {f.get('faq_id','')}",
                        f"q_keywords: {f.get('q_keywords','')}",
                        f"answer: {clamp(f.get('answer',''), 240)}",
                        "---"
                    ])
                )

        parts.append(
            "\n=== CONTACT ===\n"
            "Хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж\n"
            "Утас: 91117577, 99201187\n"
            "Имэйл: hello@wayconsulting.io\n"
        )

        return "\n".join(parts)

    def generate(self, question: str, context: str, time_budget_sec: float) -> str:
        if not self.client:
            return "Уучлаарай, AI сервис түр ажиллахгүй байна."

        sys_prompt = self.build_system_prompt()
        user_prompt = (
            f"Хэрэглэгчийн асуулт: {question}\n\n"
            f"Доорх контекстээс хариул:\n{context}\n\n"
            "Хариулт:"
        )

        # Hard cap tokens to keep latency low
        max_tokens = 450

        # NOTE: Python SDK doesn't support per-request server-side timeout uniformly.
        # We rely on overall endpoint budget + small context.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


# ======================
# Init services (fail-fast for required env)
# ======================
missing = []
for k in ["SHEET_ID", "GOOGLE_CREDENTIALS_JSON"]:
    if not app.config.get(k):
        missing.append(k)

if missing:
    logger.warning(f"⚠️ Missing env vars: {missing} (Sheets features may fail)")

sheets_service: Optional[GoogleSheetsService] = None
if app.config.get("SHEET_ID") and app.config.get("GOOGLE_CREDENTIALS_JSON"):
    sheets_service = GoogleSheetsService(
        sheet_id=app.config["SHEET_ID"],
        credentials_json_str=app.config["GOOGLE_CREDENTIALS_JSON"],
        cache_ttl=app.config["CACHE_TTL"],
    )

ai_service = AIService(api_key=app.config.get("OPENAI_API_KEY"), model=app.config["OPENAI_MODEL"])


# ======================
# Routes
# ======================
@app.get("/")
def index():
    return jsonify({
        "status": "active",
        "service": "Way Academy Chatbot API",
        "timestamp": now_iso(),
        "endpoints": {
            "/health": "Health check",
            "/manychat/webhook": "ManyChat Dynamic Block webhook (POST)",
            "/courses": "List courses (GET)",
            "/faqs": "List faqs (GET)",
        }
    })


@app.get("/health")
def health():
    status = {
        "google_sheets": bool(sheets_service),
        "openai": bool(app.config.get("OPENAI_API_KEY")),
        "cache_ttl": app.config["CACHE_TTL"],
        "model": app.config["OPENAI_MODEL"],
        "timestamp": now_iso(),
        "version": "1.0.0",
    }
    overall = "healthy" if status["google_sheets"] else "degraded"
    return jsonify({"status": overall, "services": status})


@app.post("/manychat/webhook")
def manychat_webhook():
    start = time.time()

    data = request.get_json(silent=True) or {}
    subscriber_id = data.get("subscriber_id")
    message = (data.get("message") or "").strip()

    # Basic validation (ManyChat Dynamic Block)
    if not subscriber_id or not isinstance(message, str) or not message:
        return manychat_text("Уучлаарай, таны мессежийг уншиж чадсангүй. Дахин бичнэ үү."), 200

    # Time budget guard
    budget = app.config["TIME_BUDGET_SEC"]

    try:
        if not sheets_service:
            return manychat_text("Уучлаарай, одоогоор мэдээллийн сан холбогдоогүй байна."), 200

        # 1) Pull data (cached)
        all_courses = sheets_service.get_all_courses()
        all_faqs = sheets_service.get_all_faqs()

        # 2) Match course (optional)
        matched = sheets_service.get_course_by_keyword(message)
        courses_for_ctx: List[Dict[str, Any]] = []
        if matched:
            courses_for_ctx = [matched]
        else:
            # Keep context small for speed
            courses_for_ctx = all_courses[: app.config["MAX_COURSES_IN_CONTEXT"]]

        faqs_for_ctx = all_faqs[: app.config["MAX_FAQS_IN_CONTEXT"]]

        # 3) Build context (small)
        context = ai_service.format_context(courses_for_ctx, faqs_for_ctx)

        # 4) Generate AI response if time left
        elapsed = time.time() - start
        if elapsed > budget:
            return manychat_text("Уучлаарай, систем ачаалалтай байна. Дахин оролдоно уу."), 200

        answer = ai_service.generate(message, context, budget - elapsed)
        if not answer:
            answer = "Уучлаарай, энэ асуултад одоогоор тодорхой хариулт олдсонгүй."

        # 5) Return Dynamic Block response
        return manychat_text(answer), 200

    except Exception as e:
        logger.exception(f"❌ webhook error: {e}")
        return manychat_text("Уучлаарай, техникийн алдаа гарлаа. Та дахин оролдоно уу."), 200


@app.get("/courses")
def courses():
    if not sheets_service:
        return jsonify({"count": 0, "courses": [], "error": "Sheets not configured"}), 200

    courses_data = sheets_service.get_all_courses()
    simplified = [{
        "course_id": c.get("course_id"),
        "course_name": c.get("course_name"),
        "teacher": c.get("teacher"),
        "duration": c.get("duration"),
        "schedule_1": c.get("schedule_1"),
        "price_full": c.get("price_full"),
        "price_discount": c.get("price_discount"),
        "application_link": c.get("application_link"),
        "priority": c.get("priority"),
    } for c in courses_data]

    return jsonify({"count": len(simplified), "courses": simplified})


@app.get("/faqs")
def faqs():
    if not sheets_service:
        return jsonify({"count": 0, "faqs": [], "error": "Sheets not configured"}), 200

    faqs_data = sheets_service.get_all_faqs()
    simplified = [{
        "faq_id": f.get("faq_id"),
        "q_keywords": f.get("q_keywords"),
        "answer": f.get("answer"),
        "priority": f.get("priority"),
    } for f in faqs_data]

    return jsonify({"count": len(simplified), "faqs": simplified})


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


# ======================
# Main
# ======================
if __name__ == "__main__":
    logger.info(f"🚀 Starting on :{app.config['PORT']}")
    logger.info(f"📄 SHEET_ID: {app.config.get('SHEET_ID')}")
    logger.info(f"🤖 MODEL: {app.config['OPENAI_MODEL']}")
    app.run(host="0.0.0.0", port=app.config["PORT"], debug=app.config["FLASK_DEBUG"])
