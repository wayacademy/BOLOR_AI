import os
import json
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from google.oauth2 import service_account
from googleapiclient.discovery import build

import requests
from dotenv import load_dotenv

# ✅ FIX: TTL cache ашиглах (pip install cachetools)
from cachetools import TTLCache

# ✅ FIX: OpenAI SDK зөв ашиглалт (pip install openai)
from openai import OpenAI

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======================
# Env
# ======================
load_dotenv()

app = Flask(__name__)
CORS(app)

# ======================
# Config
# ======================
class Config:
    SHEET_ID = os.getenv("SHEET_ID", "")
    GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    MANYCHAT_TOKEN = os.getenv("MANYCHAT_TOKEN", "")

    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds

app.config.from_object(Config)


# ======================
# Google Sheets Service
# ======================
class GoogleSheetsService:
    def __init__(self):
        self.sheet_id = app.config["SHEET_ID"]
        self.credentials_json = app.config["GOOGLE_CREDENTIALS_JSON"]
        self.service = None

        # ✅ FIX: 5 минут TTL cache (courses, faq гэх мэт)
        self._cache = TTLCache(maxsize=16, ttl=app.config["CACHE_TTL"])

        self._initialize_service()

    def _initialize_service(self):
        try:
            if not self.sheet_id:
                raise ValueError("SHEET_ID хоосон байна")

            if not self.credentials_json or self.credentials_json.strip() in ("", "{}"):
                raise ValueError("GOOGLE_CREDENTIALS_JSON хоосон байна")

            credentials_info = json.loads(self.credentials_json)

            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
            )

            self.service = build("sheets", "v4", credentials=credentials)
            logger.info("✅ Google Sheets API сервис эхэллээ")

        except Exception as e:
            logger.error(f"❌ Google Sheets init алдаа: {e}", exc_info=True)
            raise

    def get_sheet_data(self, sheet_name: str) -> List[Dict[str, Any]]:
        """Google Sheets-ээс хүснэгт унших (A:Z)"""
        try:
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=self.sheet_id,
                range=f"{sheet_name}!A:Z"
            ).execute()

            values = result.get("values", [])
            if not values:
                return []

            headers = values[0]
            data = []

            for row in values[1:]:
                item = {}
                for i, header in enumerate(headers):
                    item[header] = row[i] if i < len(row) else ""

                # зөвхөн is_active=True мөрүүд
                if item.get("is_active", "True").strip().lower() == "true":
                    data.append(item)

            logger.info(f"✅ {sheet_name} хуудсаас {len(data)} мөр уншлаа")
            return data

        except Exception as e:
            logger.error(f"❌ {sheet_name} унших алдаа: {e}", exc_info=True)
            return []

    def get_cached_sheet(self, sheet_name: str) -> List[Dict[str, Any]]:
        """✅ FIX: TTL cache ашиглан sheet унших"""
        if sheet_name in self._cache:
            return self._cache[sheet_name]

        data = self.get_sheet_data(sheet_name)
        self._cache[sheet_name] = data
        return data

    def get_all_courses(self) -> List[Dict[str, Any]]:
        courses = self.get_cached_sheet("courses")
        # priority өсөхөөр эрэмбэлэх
        def safe_float(x):
            try:
                return float(x)
            except:
                return 999.0
        return sorted(courses, key=lambda x: safe_float(x.get("priority", 999)))

    def get_all_faqs(self) -> List[Dict[str, Any]]:
        return self.get_cached_sheet("faq")

    def get_course_by_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Түлхүүр үг/нэр/ID-гаар 1 course тааруулах"""
        if not keyword:
            return None

        keyword_lower = keyword.strip().lower()
        courses = self.get_all_courses()

        for course in courses:
            # keywords column: "a|b|c"
            keywords = (course.get("keywords", "") or "").strip()
            if keywords and "|" in keywords:
                course_keywords = [k.strip().lower() for k in keywords.split("|") if k.strip()]
                if any(kw in keyword_lower for kw in course_keywords):
                    return course

            course_name = (course.get("course_name", "") or "").strip().lower()
            if course_name and keyword_lower in course_name:
                return course

            course_id = (course.get("course_id", "") or "").strip().lower()
            if course_id and keyword_lower == course_id:
                return course

        return None


# ======================
# AI Service
# ======================
class AIService:
    def __init__(self):
        self.api_key = app.config["OPENAI_API_KEY"]
        self.model = app.config["OPENAI_MODEL"]
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY тохируулаагүй байна")

    def generate_response(self, user_question: str, context_data: Dict[str, Any]) -> str:
        if not self.client:
            return "Уучлаарай, AI сервис түр ажиллахгүй байна. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."

        system_prompt = (
            "Та бол Way Academy-гийн албан ёсны туслах чатбот.\n"
            "Дараах дүрмийг баримтлаарай:\n"
            "1) ЗӨВХӨН өгөгдсөн мэдээллээс хариул\n"
            "2) Монгол хэлээр, найрсаг, товч\n"
            "3) Үнэ, цаг, багшийн мэдээллийг тодорхой харуул\n"
            "4) Мэдээлэл олдохгүй бол: \"Уучлаарай, энэ асуултанд хариулж чадахгүй байна\" гэж хэл\n"
            "5) Сургалтын нэр, ID (SDM, DA гэх мэт)-г зөв хэрэглэ\n"
            "6) Төгсгөлд: \"Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу\" гэж нэм"
        )

        context_str = self._format_context(context_data)

        user_prompt = (
            f"Хэрэглэгчийн асуулт: {user_question}\n\n"
            "Доорх мэдээллээс хариулт өгнө үү:\n"
            f"{context_str}\n\n"
            "Хариулт:"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=450
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                return "Уучлаарай, хариулт үүсгэхэд алдаа гарлаа. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."
            return text

        except Exception as e:
            logger.error(f"❌ OpenAI алдаа: {e}", exc_info=True)
            return self._fallback(user_question, context_data)

    def _format_context(self, data: Dict[str, Any]) -> str:
        parts = []

        courses = data.get("courses", [])
        if courses:
            parts.append("=== СУРГАЛТУУД ===")
            for c in courses:
                parts.append(
                    "Нэр: {name}\n"
                    "ID: {cid}\n"
                    "Тайлбар: {desc}\n"
                    "Багш: {teacher}\n"
                    "Хугацаа: {duration}\n"
                    "Үнэ: {price_full}\n"
                    "Early Bird: {price_early} ({early_note})\n"
                    "Цагийн хуваарь: {s1} {s2}\n"
                    "---".format(
                        name=c.get("course_name", "Тодорхойгүй"),
                        cid=c.get("course_id", "Тодорхойгүй"),
                        desc=(c.get("description", "") or "")[:200] + ("..." if c.get("description") else ""),
                        teacher=c.get("teacher", "Тодорхойгүй"),
                        duration=c.get("duration", "Тодорхойгүй"),
                        price_full=c.get("price_full", "Тодорхойгүй"),
                        # таны sheet-д price_early_bird/early_bird_note байгаа гэж санаж байна
                        price_early=c.get("price_early_bird", c.get("price_discount", "Байхгүй")),
                        early_note=c.get("early_bird_note", c.get("price_discount_until", "")),
                        s1=c.get("schedule_1", ""),
                        s2=c.get("schedule_2", "")
                    )
                )

        faqs = data.get("faqs", [])
        if faqs:
            parts.append("\n=== ТҮГЭЭМЭЛ АСУУЛТУУД ===")
            for f in faqs:
                parts.append(
                    "Түлхүүр: {k}\n"
                    "Хариулт: {a}\n"
                    "---".format(
                        k=f.get("q_keywords", ""),
                        a=(f.get("answer", "") or "")[:200] + ("..." if f.get("answer") else "")
                    )
                )

        parts.append(
            "\n=== БУСАД МЭДЭЭЛЭЛ ===\n"
            "Хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж\n"
            "Утас: 91117577, 99201187\n"
            "Имэйл: hello@wayconsulting.io\n"
            "Академийн онцлог: Салбарын шилдэг багш нар, Бодит төсөл дээр практик, AI-г сургалтад нэвтрүүлсэн"
        )

        return "\n".join(parts)

    def _fallback(self, user_question: str, context_data: Dict[str, Any]) -> str:
        # Маш энгийн fallback
        q = (user_question or "").lower()
        if "хаяг" in q:
            return "Манай хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."
        if "утас" in q or "дугаар" in q:
            return "Утас: 91117577, 99201187. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."
        return "Уучлаарай, энэ асуултанд хариулж чадахгүй байна. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."


# ======================
# Initialize services
# ======================
try:
    sheets_service = GoogleSheetsService()
except Exception:
    sheets_service = None

ai_service = AIService()


# ======================
# Helpers
# ======================
def parse_manychat_payload(data: Dict[str, Any]) -> (Optional[str], str):
    """
    ManyChat payload олон янзаар ирдэг.
    subscriber_id + user_message-г аль болох найдвартай гаргаж авна.
    """
    subscriber_id = None
    user_message = ""

    # 1) { subscriber_id, message: "text" }
    if "subscriber_id" in data:
        subscriber_id = str(data.get("subscriber_id"))
        msg = data.get("message", "")
        user_message = msg.get("text", "") if isinstance(msg, dict) else str(msg or "")

    # 2) { subscriber: {id}, message: {text} }
    elif "subscriber" in data:
        sub = data.get("subscriber") or {}
        subscriber_id = str(sub.get("id")) if sub.get("id") is not None else None
        msg = data.get("message", {})
        if isinstance(msg, dict):
            user_message = msg.get("text", "") or msg.get("message", "") or ""
        else:
            user_message = str(msg or "")

    # 3) { data: { subscriber: {id}, message: ... } }
    elif "data" in data and isinstance(data["data"], dict):
        inner = data["data"]
        sub = inner.get("subscriber") or {}
        subscriber_id = str(sub.get("id")) if sub.get("id") is not None else None
        msg = inner.get("message", "")
        user_message = msg.get("text", "") if isinstance(msg, dict) else str(msg or "")

    # fallback
    if not user_message:
        user_message = (data.get("text") or data.get("message") or "").strip()

    return subscriber_id, (user_message or "").strip()


# ======================
# Routes
# ======================
@app.route("/")
def index():
    return jsonify({
        "status": "active",
        "service": "Way Academy Chatbot API",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "endpoints": {
            "/health": "health",
            "/manychat/webhook": "ManyChat webhook",
            "/test": "local test",
            "/courses": "list courses",
            "/faqs": "list faqs"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "google_sheets": bool(sheets_service),
        "openai": bool(app.config["OPENAI_API_KEY"]),
        "timestamp": datetime.now().isoformat()
    })


@app.route("/manychat/webhook", methods=["POST"])
def manychat_webhook():
    try:
        data = request.get_json(silent=True) or {}
        subscriber_id, user_message = parse_manychat_payload(data)

        if not subscriber_id:
            logger.warning(f"❌ Subscriber ID олдсонгүй. keys={list(data.keys())}")
            return jsonify({"error": "Subscriber ID олдсонгүй"}), 400

        if not user_message:
            user_message = "сайн уу"

        logger.info(f"📩 ManyChat message: {user_message[:80]}... (Subscriber: {subscriber_id})")

        # Sheets data
        all_courses, all_faqs = [], []
        if sheets_service:
            all_courses = sheets_service.get_all_courses()
            all_faqs = sheets_service.get_all_faqs()
        else:
            logger.error("❌ Google Sheets сервис ажиллахгүй байна")

        # Match course unless greeting
        greetings = {"сайн уу", "сайн байна уу", "hello", "hi", "сайн", "байна уу"}
        matched_courses = []
        if user_message.strip().lower() not in greetings and sheets_service:
            course = sheets_service.get_course_by_keyword(user_message)
            if course:
                matched_courses = [course]

        context_data = {
            "courses": matched_courses if matched_courses else all_courses[:4],
            "faqs": all_faqs[:5]
        }

        ai_response = ai_service.generate_response(user_message, context_data)

        # ✅ FIX: ManyChat External Request + Response mapping-т зориулсан ГАНЦ формат
        # JSONPath: $.content.messages[0].text
        return jsonify({
            "version": "v2",
            "content": {
                "messages": [{
                    "type": "text",
                    "text": ai_response
                }]
            }
        })

    except Exception as e:
        logger.error(f"❌ Webhook алдаа: {e}", exc_info=True)
        return jsonify({
            "version": "v2",
            "content": {
                "messages": [{
                    "type": "text",
                    "text": "Уучлаарай, техникийн алдаа гарлаа. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."
                }]
            }
        }), 500


@app.route("/test", methods=["POST", "GET"])
def test():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "дижитал маркетинг сургалт").strip()

        courses = sheets_service.get_all_courses() if sheets_service else []
        faqs = sheets_service.get_all_faqs() if sheets_service else []

        context = {"courses": courses[:2], "faqs": faqs[:2]}
        answer = ai_service.generate_response(question, context)

        return jsonify({
            "question": question,
            "answer": answer,
            "courses_count": len(courses),
            "faqs_count": len(faqs)
        })

    # GET
    courses = sheets_service.get_all_courses() if sheets_service else []
    faqs = sheets_service.get_all_faqs() if sheets_service else []
    return jsonify({
        "courses_count": len(courses),
        "faqs_count": len(faqs),
        "sample_course": courses[0].get("course_name") if courses else None,
        "sample_faq": faqs[0].get("q_keywords") if faqs else None
    })


@app.route("/courses", methods=["GET"])
def courses():
    courses = sheets_service.get_all_courses() if sheets_service else []
    simplified = [{
        "id": c.get("course_id"),
        "name": c.get("course_name"),
        "teacher": c.get("teacher"),
        "duration": c.get("duration"),
        "price_full": c.get("price_full"),
        "price_early_bird": c.get("price_early_bird"),
        "early_bird_note": c.get("early_bird_note"),
        "schedule_1": c.get("schedule_1"),
    } for c in courses]
    return jsonify({"count": len(simplified), "courses": simplified})


@app.route("/faqs", methods=["GET"])
def faqs():
    faqs_ = sheets_service.get_all_faqs() if sheets_service else []
    simplified = [{
        "id": f.get("faq_id"),
        "q_keywords": f.get("q_keywords"),
        "answer": f.get("answer"),
    } for f in faqs_]
    return jsonify({"count": len(simplified), "faqs": simplified})


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint олдсонгүй"}), 404


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Дотоод серверийн алдаа"}), 500


if __name__ == "__main__":
    # ✅ FIX: production дээр заавал байх env-үүд
    required = ["SHEET_ID", "GOOGLE_CREDENTIALS_JSON"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error(f"❌ Дараах env дутуу байна: {missing}")
        # Railway дээр унах нь зөв (алдаатай ажиллуулахгүй)
        sys.exit(1)

    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"🚀 Server starting on :{port} (debug={debug_mode})")
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)
