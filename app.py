import os
import json
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

from google.oauth2 import service_account
from googleapiclient.discovery import build
from cachetools import TTLCache

# OpenAI-г хасаж, Google Generative AI-г оруулж ирэв
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError


# ======================
# Logging
# ======================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
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
    SHEET_ID = os.getenv("SHEET_ID", "").strip()
    GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

    # Cache
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds

    # Gemini Config (Updated)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    # "2.5 flash" гэж байхгүй тул одоогоор хамгийн сүүлийн stable хувилбар болох 1.5-flash-ийг сонгов.
    # Хэрэв 2.0 гарсан бол 'gemini-2.0-flash-exp' гэж сольж болно.
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    # ManyChat time budget (~10s). Keep our budget lower.
    TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "8.5"))

    # Context limits
    MAX_COURSES_IN_CONTEXT = int(os.getenv("MAX_COURSES_IN_CONTEXT", "20"))
    MAX_FAQS_IN_CONTEXT = int(os.getenv("MAX_FAQS_IN_CONTEXT", "20"))
    MAX_DESC_CHARS = int(os.getenv("MAX_DESC_CHARS", "260"))

    # Dedup (idempotency)
    DEDUP_TTL_SEC = int(os.getenv("DEDUP_TTL_SEC", "30"))
    DEDUP_MAXSIZE = int(os.getenv("DEDUP_MAXSIZE", "5000"))

    # Template response limits
    MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "1200"))


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


def clamp(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n].rstrip() + "..."

def normalize_answer(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"[*_`#]", "", t)          # Markdown арилгана
    t = re.sub(r"\n\s*-\s*$", "", t)      # сүүлчийн дан '-' мөрийг авна
    return t


def manychat_v2(text: str):
    """Dynamic Block (v2) response."""
    # ManyChat sometimes displays URLs better than markdown; keep plain text.
    text = (text or "").strip()
    if len(text) > app.config["MAX_TEXT_CHARS"]:
        text = text[: app.config["MAX_TEXT_CHARS"]].rstrip() + "..."
    return jsonify(
        {
            "version": "v2",
            "content": {"messages": [{"type": "text", "text": text}]},
        }
    )


def manychat_empty():
    """Return nothing (used for dedup)."""
    return jsonify({"version": "v2", "content": {"messages": []}})


def safe_json() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}


def extract_manychat_fields(payload: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    External Request (Dynamic Block) body should send:
      subscriber_id, message
    But we defensively check a few alternatives.
    """
    subscriber_id = payload.get("subscriber_id") or payload.get("contact_id") or payload.get("subscriberId")
    msg = payload.get("message") or payload.get("last_text_input") or payload.get("last_input_text") or ""
    if not isinstance(msg, str):
        msg = str(msg)
    return (str(subscriber_id).strip() if subscriber_id else None), msg.strip()


# ======================
# Dedup cache (idempotency)
# ======================
dedup_cache = TTLCache(maxsize=app.config["DEDUP_MAXSIZE"], ttl=app.config["DEDUP_TTL_SEC"])


# ======================
# Google Sheets Service (TTL Cache)
# ======================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


class GoogleSheetsService:
    def __init__(self, sheet_id: str, credentials_json_str: str, cache_ttl: int = 300):
        self.sheet_id = sheet_id
        self.cache = TTLCache(maxsize=32, ttl=cache_ttl)
        self.service = self._init_service(credentials_json_str)

    def _init_service(self, credentials_json_str: str):
        info = json.loads(credentials_json_str)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        # cache_discovery=False speeds startup and avoids file writes on some hosts
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

        try:
            values = self._read_values(sheet_name)
        except Exception as e:
            logger.exception(f"❌ Sheets read error ({sheet_name}): {e}")
            self.cache[cache_key] = []
            return []

        if not values:
            self.cache[cache_key] = []
            return []

        headers = values[0]
        out: List[Dict[str, Any]] = []

        for row in values[1:]:
            item = {h: (row[i] if i < len(row) else "") for i, h in enumerate(headers)}
            is_active = str(item.get("is_active", "True")).strip().lower() == "true"
            if is_active:
                out.append(item)

        self.cache[cache_key] = out
        logger.info(f"✅ Loaded {len(out)} rows from '{sheet_name}' (cached)")
        return out

    def get_all_faqs(self) -> List[Dict[str, Any]]:
        return self.get_sheet_dicts("faq")

    def get_all_courses(self) -> List[Dict[str, Any]]:
        return self.get_sheet_dicts("courses")

    def get_course_by_keyword(self, user_text: str) -> Optional[Dict[str, Any]]:
        t = (user_text or "").lower().strip()
        if not t:
            return None

        courses = self.get_all_courses()

        for c in courses:
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
# AI Service (Gemini)
# ======================
class AIService:
    def __init__(self, api_key: str, model: str):
        self.model_name = model
        self.api_key = api_key
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # System prompt-ийг энд тодорхойлох нь илүү үр дүнтэй (Gemini GenerativeModel config)
            self.system_instruction = self.build_system_prompt()
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction
            )
        else:
            self.model = None
            logger.warning("⚠️ GEMINI_API_KEY missing; AI disabled")

    def build_system_prompt(self) -> str:
        return (
            "Та бол Way Academy-гийн албан ёсны зөвлөх чатбот.\n"
            "Зорилго: Хэрэглэгчийн асуултад өгөгдсөн контекстээс хариулах.\n"
            "\n"
            "Өгөгдөл ашиглах заавар:\n"
            "1) ЗӨВХӨН доор өгөгдсөн 'COURSES' болон 'FAQ' хэсгээс мэдээллийг авч хариул. Гаднаас зохиож болохгүй.\n"
            "2) Хэрэв хэрэглэгч тодорхой хөтөлбөрийн нэр дурдаагүй ч 'багш', 'үнэ', 'хугацаа' зэргийг асуувал:\n"
            "   - Бүх хөтөлбөрийн тухайн мэдээллийг товч жагсааж бич.\n"
            "   - Жишээ нь: 'DA хөтөлбөрийн багш ..., харин SDM хөтөлбөрийн багш ...'\n"
            "3) Markdown тэмдэгт ашиглахгүй. Энгийн текстээр бич.\n"
            "\n"
            "Хариулах хэлбэр:\n"
            "- Хэрэв хэрэглэгч 'Сайн байна уу', 'ямар сургалт байна' гэх мэт ерөнхий асуувал:\n"
            "  Сайн байна уу? Бид дараах эрэлттэй хөтөлбөрүүдийг санал болгож байна:\n"
            "  - Стратегийн дижитал маркетинг (SDM)\n"
            "  - Дата аналист (DA)\n"
            "  - IT Бизнес шинжээч (ITBA)\n"
            "  - Project Zero: AI Agent Developer (PZ)\n"
            "  Та алийг нь сонирхож байна вэ?\n"
            "- Хэрэв хэрэглэгч тодорхой асуулт (багш, үнэ г.м) асуувал шууд хариултыг нь өг.\n"
        )

    def format_context(self, courses: List[Dict[str, Any]], faqs: List[Dict[str, Any]]) -> str:
        parts: List[str] = []

        if courses:
            parts.append("=== COURSES ===")
            for c in courses:
                parts.append(
                    "\n".join(
                        [
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
                            "---",
                        ]
                    )
                )

        if faqs:
            parts.append("\n=== FAQ ===")
            for f in faqs:
                parts.append(
                    "\n".join(
                        [
                            f"faq_id: {f.get('faq_id','')}",
                            f"q_keywords: {f.get('q_keywords','')}",
                            f"answer: {clamp(f.get('answer',''), 240)}",
                            "---",
                        ]
                    )
                )

        parts.append(
            "\n=== CONTACT ===\n"
            "Хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж\n"
            "Утас: 91117577, 99201187\n"
            "Имэйл: hello@wayconsulting.io\n"
        )

        return "\n".join(parts)

    def generate(self, question: str, context: str) -> str:
        if not self.model:
            return "Уучлаарай, AI сервис түр ажиллахгүй байна."

        try:
            # Gemini-д зориулсан prompt бүтэц
            full_prompt = (
                f"Хэрэглэгчийн асуулт: {question}\n\n"
                f"Доорх контекстээс хариул:\n{context}\n\n"
                f"Хариулт:"
            )

            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.35,
                    max_output_tokens=420,
                )
            )
            
            return (response.text or "").strip()
        
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return "Уучлаарай, системд алдаа гарлаа. Та дараа дахин оролдоно уу."


# ======================
# Init services
# ======================
sheets_service: Optional[GoogleSheetsService] = None
if app.config["SHEET_ID"] and app.config["GOOGLE_CREDENTIALS_JSON"]:
    try:
        sheets_service = GoogleSheetsService(
            sheet_id=app.config["SHEET_ID"],
            credentials_json_str=app.config["GOOGLE_CREDENTIALS_JSON"],
            cache_ttl=app.config["CACHE_TTL"],
        )
    except Exception as e:
        logger.exception(f"❌ Failed to init Sheets: {e}")
        sheets_service = None
else:
    logger.warning("⚠️ SHEET_ID / GOOGLE_CREDENTIALS_JSON missing")

ai_service = AIService(api_key=app.config["GEMINI_API_KEY"], model=app.config["GEMINI_MODEL"])


# ======================
# Fast template response (no AI) for matched course
# ======================
def format_course_template(c: Dict[str, Any]) -> str:
    name = c.get("course_name", "Тодорхойгүй")
    price_full = c.get("price_full", "")
    price_disc = c.get("price_discount", "")
    disc_until = c.get("price_discount_until", "")
    duration = c.get("duration", "")
    teacher = c.get("teacher", "")
    s1 = c.get("schedule_1", "")
    s2 = c.get("schedule_2", "")
    pay = c.get("payment_options", "")
    link = c.get("application_link", "")
    cta = c.get("cta_caption", "")

    lines = [
        f"{name}",
        f"Үнэ: {price_full}" if price_full else "Үнэ: (мэдээлэл алга)",
    ]
    if price_disc:
        extra = f"Early Bird: {price_disc}"
        if disc_until:
            extra += f" (Хугацаа: {disc_until})"
        lines.append(extra)

    if duration:
        lines.append(f"Хугацаа: {duration}")
    if teacher:
        lines.append(f"Багш: {teacher}")
    if s1 or s2:
        lines.append("Цагийн хуваарь:")
        if s1:
            lines.append(f"- {s1}")
        if s2:
            lines.append(f"- {s2}")
    if pay:
        lines.append(f"Төлбөрийн нөхцөл: {pay}")
    if link:
        lines.append(f"Бүртгүүлэх: {link}")
    if cta:
        lines.append(cta)

    return "\n".join([ln for ln in lines if ln and ln.strip()])


# ======================
# Routes
# ======================
@app.get("/")
def index():
    return jsonify(
        {
            "status": "active",
            "service": "Way Academy Chatbot API (Gemini)",
            "timestamp": now_iso(),
            "endpoints": {
                "/health": "Health check",
                "/manychat/webhook": "ManyChat Dynamic Block webhook (POST)",
                "/courses": "List courses (GET)",
                "/faqs": "List faqs (GET)",
            },
        }
    )


@app.get("/health")
def health():
    services = {
        "google_sheets": bool(sheets_service),
        "gemini": bool(app.config["GEMINI_API_KEY"]),
        "cache_ttl": app.config["CACHE_TTL"],
        "model": app.config["GEMINI_MODEL"],
        "timestamp": now_iso(),
        "version": "1.2.0-gemini",
        "dedup_ttl": app.config["DEDUP_TTL_SEC"],
    }
    overall = "healthy" if services["google_sheets"] else "degraded"
    return jsonify({"status": overall, "services": services})


@app.post("/manychat/webhook")
def manychat_webhook():
    start = time.time()

    payload = request.get_json(silent=True) or {}
    subscriber_id = payload.get("subscriber_id")
    message = (payload.get("message") or "").strip()

    logger.info(f"[MC] subscriber_id={subscriber_id} message={message!r}")

    # Validate
    if not subscriber_id or not message:
        return jsonify({"ai_response_text": "Уучлаарай, таны мессежийг уншиж чадсангүй. Дахин бичнэ үү."}), 200

    # Dedup
    key = f"{subscriber_id}:{message}"
    if key in dedup_cache:
        logger.info(f"[MC] dedup hit: {key}")
        return jsonify({"ai_response_text": ""}), 200
    dedup_cache[key] = True

    if not sheets_service:
        return jsonify({"ai_response_text": "Уучлаарай, одоогоор мэдээллийн сан холбогдоогүй байна."}), 200

    try:
        # 1. Sheet-ээс бүх мэдээллийг татах
        all_courses = sheets_service.get_all_courses()
        all_faqs = sheets_service.get_all_faqs()

        # Time budget guard
        if (time.time() - start) > app.config["TIME_BUDGET_SEC"]:
            return jsonify({"ai_response_text": "Уучлаарай, систем ачаалалтай байна. Дахин оролдоно уу."}), 200

        # Gemini Flash нь context window томтой тул бүх мэдээллийг өгч болно.
        context = ai_service.format_context(all_courses, all_faqs) 

        # AI
        answer = ai_service.generate(message, context)
        
        if not answer:
            answer = "Уучлаарай, энэ асуултад одоогоор тодорхой хариулт олдсонгүй."
            
        answer = normalize_answer(answer)

        return jsonify({"ai_response_text": answer}), 200

    except Exception as e:
        logger.exception(f"❌ webhook error: {e}")
        return jsonify({"ai_response_text": "Уучлаарай, техникийн алдаа гарлаа. Та дахин оролдоно уу."}), 200


@app.get("/courses")
def courses():
    if not sheets_service:
        return jsonify({"count": 0, "courses": [], "error": "Sheets not configured"}), 200

    courses_data = sheets_service.get_all_courses()
    simplified = [
        {
            "course_id": c.get("course_id"),
            "course_name": c.get("course_name"),
            "teacher": c.get("teacher"),
            "duration": c.get("duration"),
            "schedule_1": c.get("schedule_1"),
            "price_full": c.get("price_full"),
            "price_discount": c.get("price_discount"),
            "application_link": c.get("application_link"),
            "priority": c.get("priority"),
        }
        for c in courses_data
    ]
    return jsonify({"count": len(simplified), "courses": simplified})


@app.get("/faqs")
def faqs():
    if not sheets_service:
        return jsonify({"count": 0, "faqs": [], "error": "Sheets not configured"}), 200

    faqs_data = sheets_service.get_all_faqs()
    simplified = [
        {
            "faq_id": f.get("faq_id"),
            "q_keywords": f.get("q_keywords"),
            "answer": f.get("answer"),
            "priority": f.get("priority"),
        }
        for f in faqs_data
    ]
    return jsonify({"count": len(simplified), "faqs": simplified})


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


# ======================
# Main (local only)
# ======================
if __name__ == "__main__":
    logger.info(f"🚀 Starting on :{app.config['PORT']}")
    logger.info(f"📄 SHEET_ID: {app.config.get('SHEET_ID')}")
    logger.info(f"🤖 MODEL: {app.config['GEMINI_MODEL']}")
    app.run(host="0.0.0.0", port=app.config["PORT"], debug=app.config["FLASK_DEBUG"])