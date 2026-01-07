import os
import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai
from dotenv import load_dotenv

# ======================
# Лог тохируулах
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables ачаалах
load_dotenv()

app = Flask(__name__)
CORS(app)

# ======================
# Конфигураци
# ======================
class Config:
    # Google Sheets API тохиргоо
    SHEET_ID = os.getenv("SHEET_ID", "1HG2o-2oJtYwCWoGQpC3HhC_n6_scR-cPrMB47U9yc90")
    CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "{}")

    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Кэш хугацаа (секундээр) - LRU cache TTL биш гэдгийг санаарай
    CACHE_TTL = 300  # 5 минут (одоогоор зөвхөн “concept”)

app.config.from_object(Config)

# ======================
# Google Sheets Service
# ======================
class GoogleSheetsService:
    def __init__(self):
        self.sheet_id = app.config["SHEET_ID"]
        self.service = None
        self._initialize_service()

    def _initialize_service(self):
        """Google Sheets API сервис эхлүүлэх"""
        try:
            credentials_raw = app.config["CREDENTIALS_JSON"]
            credentials_info = json.loads(credentials_raw)

            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
            )
            self.service = build("sheets", "v4", credentials=credentials)
            logger.info("✅ Google Sheets API сервис эхлэв")
        except Exception as e:
            logger.error(f"❌ Google Sheets API эхлүүлэхэд алдаа: {e}")
            raise

    @lru_cache(maxsize=2)
    def get_cached_data(self, sheet_name: str, cache_key: str = ""):
        """
        ⚠️ REMINDER: lru_cache нь TTL биш.
        Одоогоор “давтан уншихыг багасгах” л үүрэгтэй.
        TTL хэрэгтэй бол cachetools (TTLCache) ашиглана.
        """
        return self.get_sheet_data(sheet_name)

    def get_sheet_data(self, sheet_name: str) -> List[Dict[str, Any]]:
        """Google Sheets-ээс өгөгдөл унших"""
        try:
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=self.sheet_id,
                range=f"{sheet_name}!A:Z",
            ).execute()

            values = result.get("values", [])
            if not values:
                return []

            headers = values[0]
            data: List[Dict[str, Any]] = []

            for row in values[1:]:
                item = {}
                for i, header in enumerate(headers):
                    item[header] = row[i] if i < len(row) else ""

                if item.get("is_active", "True").strip().lower() == "true":
                    data.append(item)

            logger.info(f"✅ {sheet_name} хуудаснаас {len(data)} мөр уншлаа")
            return data

        except Exception as e:
            logger.error(f"❌ {sheet_name} хуудсыг уншихад алдаа: {e}")
            return []

    def get_all_courses(self) -> List[Dict[str, Any]]:
        courses = self.get_cached_data("courses", "courses_cache")
        return sorted(courses, key=lambda x: float(x.get("priority", 999)))

    def get_all_faqs(self) -> List[Dict[str, Any]]:
        return self.get_cached_data("faq", "faq_cache")

    def get_course_by_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        if not keyword:
            return None

        keyword_lower = keyword.lower().strip()
        courses = self.get_all_courses()

        for course in courses:
            keywords = course.get("keywords", "")
            if keywords and "|" in keywords:
                course_keywords = [k.strip().lower() for k in keywords.split("|")]
                if any(kw in keyword_lower for kw in course_keywords):
                    return course

            course_name = course.get("course_name", "").lower()
            if keyword_lower in course_name:
                return course

        return None

# ======================
# AI Service (OpenAI)
# ======================
class AIService:
    def __init__(self):
        openai.api_key = app.config["OPENAI_API_KEY"]
        self.model = app.config["OPENAI_MODEL"]

        if not openai.api_key:
            logger.warning("⚠️ OpenAI API Key олдсонгүй!")

    def generate_response(self, user_question: str, context_data: Dict[str, Any]) -> str:
        try:
            if not openai.api_key:
                return "Уучлаарай, AI сервис түр ажиллахгүй байна."

            system_prompt = """Та бол Way Academy-гийн албан ёсны туслах чатбот.
Дараах дүрмийг баримтлаарай:
1. ЗӨВХӨН өгөгдсөн мэдээллээс хариулт өгөх
2. Монгол хэлээр, найрсаг, товч хариулт өгөх
3. Үнэ, цаг, багшийн мэдээллийг тодорхой харуулах
4. Хэрэв мэдээлэл олдохгүй бол "Уучлаарай, энэ асуултанд хариулж чадахгүй байна" гэж хэлэх"""

            context_str = self._format_context(context_data)

            user_prompt = f"""Хэрэглэгчийн асуулт: {user_question}

Доорх мэдээллээс хариулт өгнө үү:
{context_str}

Хариулт:"""

            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"❌ AI хариулт үүсгэхэд алдаа: {e}")
            return "Уучлаарай, хариулт үүсгэхэд алдаа гарлаа. Дахин оролдоно уу."

    def _format_context(self, data: Dict[str, Any]) -> str:
        context_parts = []

        if data.get("courses"):
            context_parts.append("=== СУРГАЛТУУД ===")
            for course in data["courses"]:
                context_parts.append(f"""
Нэр: {course.get('course_name', 'Тодорхойгүй')}
ID: {course.get('course_id', 'Тодорхойгүй')}
Тайлбар: {course.get('description', '')[:200]}...
Багш: {course.get('teacher', 'Тодорхойгүй')}
Хугацаа: {course.get('duration', 'Тодорхойгүй')}
Үнэ: {course.get('price_full', 'Тодорхойгүй')}
Early Bird: {course.get('price_discount', 'Байхгүй')} ({course.get('price_discount_until', '')})
Цагийн хуваарь: {course.get('schedule_1', '')} {course.get('schedule_2', '')}
Түлхүүр үгс: {course.get('keywords', '')}
---""")

        if data.get("faqs"):
            context_parts.append("\n=== ТҮГЭЭМЭЛ АСУУЛТУУД ===")
            for faq in data["faqs"]:
                context_parts.append(f"""
Асуулт: {faq.get('q_keywords', '')}
Хариулт: {faq.get('answer', '')[:150]}...
---""")

        context_parts.append("""
=== БУСАД МЭДЭЭЛЭЛ ===
Хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж
Утас: 91117577, 99201187
Имэйл: hello@wayconsulting.io
Академийн онцлог: Салбарын шилдэг багш нар, Бодит төсөл дээр практик, AI-г сургалтад нэвтрүүлсэн
""")

        return "\n".join(context_parts)

# ======================
# Helpers: ManyChat payload parsing
# ======================
def _extract_manychat_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    ✅ FIX: ManyChat External Request payload олон янзаар ирж болдог тул
    subscriber_id + message-ийг олон хувилбараас олборлоно.
    """
    subscriber_id = None
    message_text = ""

    # Common shape 1: { subscriber_id, message }
    if isinstance(payload.get("subscriber_id"), (str, int)):
        subscriber_id = str(payload.get("subscriber_id"))
        msg = payload.get("message")
        if isinstance(msg, dict):
            message_text = str(msg.get("text", "")).strip()
        else:
            message_text = str(msg or "").strip()

    # Common shape 2: { subscriber: {id}, message: {text} }
    if not subscriber_id and isinstance(payload.get("subscriber"), dict):
        if payload["subscriber"].get("id") is not None:
            subscriber_id = str(payload["subscriber"].get("id"))
        msg = payload.get("message")
        if isinstance(msg, dict):
            message_text = str(msg.get("text", "")).strip()
        else:
            message_text = str(msg or "").strip()

    # Common shape 3: { data: { subscriber: {id}, message: ... } }
    if not subscriber_id and isinstance(payload.get("data"), dict):
        data = payload["data"]
        if isinstance(data.get("subscriber"), dict) and data["subscriber"].get("id") is not None:
            subscriber_id = str(data["subscriber"].get("id"))
        msg = data.get("message")
        if isinstance(msg, dict):
            message_text = str(msg.get("text", "")).strip()
        else:
            message_text = str(msg or "").strip()

    # Fallback
    if not message_text:
        message_text = str(payload.get("text") or payload.get("message") or "").strip()

    if not message_text:
        message_text = "сайн уу"

    return subscriber_id, message_text

# ======================
# Глобал сервисүүд
# ======================
sheets_service = GoogleSheetsService()
ai_service = AIService()

# ======================
# Flask Routes
# ======================
@app.route("/")
def index():
    return jsonify({
        "status": "active",
        "service": "Way Academy Chatbot API",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/health": "Эрүүл мэндийн шалгалт",
            "/manychat/webhook": "ManyChat вебхук",
            "/test": "Тестийн endpoint",
            "/courses": "Бүх сургалтууд",
            "/faqs": "Бүх FAQ",
        },
    })

@app.route("/health", methods=["GET"])
def health_check():
    services_status = {
        "google_sheets": False,
        "openai": bool(app.config["OPENAI_API_KEY"]),
    }

    try:
        test_data = sheets_service.get_all_courses()
        services_status["google_sheets"] = len(test_data) > 0
    except Exception:
        services_status["google_sheets"] = False

    return jsonify({
        "status": "healthy" if all(services_status.values()) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": services_status,
        "version": "1.1.0",
    })

@app.route("/manychat/webhook", methods=["POST"])
def manychat_webhook():
    """
    ✅ FIX: ManyChat External Request дээрх mapping чинь:
      JSONPath: $.content.messages[0].text
    Тиймээс бид response-г яг энэ хэлбэрээр буцаана.

    ⚠️ REMINDER: Энэ webhook дотор ManyChat API руу sendContent хийхгүй.
    (Тэгвэл давхар мессеж / 400 error / flow эвдрэх эрсдэлтэй.)
    """
    try:
        payload = request.get_json(silent=True) or {}
        subscriber_id, user_message = _extract_manychat_payload(payload)

        logger.info(f"📩 ManyChat ирсэн мессеж: {user_message[:80]}... (Subscriber: {subscriber_id})")

        # 1) Sheets-ээс мэдээлэл авах
        all_courses = sheets_service.get_all_courses()
        all_faqs = sheets_service.get_all_faqs()

        # 2) Course match
        matched_courses = []
        if user_message:
            course = sheets_service.get_course_by_keyword(user_message)
            if course:
                matched_courses = [course]

        # 3) AI response
        context_data = {
            "courses": matched_courses if matched_courses else all_courses[:4],
            "faqs": all_faqs[:5],
        }
        ai_response = ai_service.generate_response(user_message, context_data)

        # 4) ManyChat Response mapping-д зориулж яг зөв бүтэцээр буцаах
        return jsonify({
            "content": {
                "messages": [{
                    "type": "text",
                    "text": ai_response
                }]
            }
        })

    except Exception as e:
        logger.error(f"❌ Вебхук боловсруулахад алдаа: {e}", exc_info=True)
        return jsonify({
            "content": {
                "messages": [{
                    "type": "text",
                    "text": "Уучлаарай, техникийн алдаа гарлаа. Та дахин оролдоно уу эсвэл 91117577 дугаарт залгана уу."
                }]
            }
        }), 200

@app.route("/test", methods=["GET", "POST"])
def test_endpoint():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        question = data.get("question", "дижитал маркетинг сургалт")

        courses = sheets_service.get_all_courses()
        faqs = sheets_service.get_all_faqs()

        context = {"courses": courses[:2], "faqs": faqs[:2]}
        response = ai_service.generate_response(question, context)

        return jsonify({
            "question": question,
            "ai_response": response,
            "courses_count": len(courses),
            "faqs_count": len(faqs),
        })

    courses = sheets_service.get_all_courses()
    faqs = sheets_service.get_all_faqs()

    return jsonify({
        "total_courses": len(courses),
        "total_faqs": len(faqs),
        "sample_course": courses[0].get("course_name") if courses else None,
        "sample_faq": faqs[0].get("q_keywords") if faqs else None,
    })

@app.route("/courses", methods=["GET"])
def get_courses():
    courses = sheets_service.get_all_courses()
    simplified = [{
        "id": c.get("course_id"),
        "name": c.get("course_name"),
        "teacher": c.get("teacher"),
        "duration": c.get("duration"),
        "price": c.get("price_full"),
        "discount": c.get("price_discount"),
        "schedule": c.get("schedule_1"),
    } for c in courses]

    return jsonify({"count": len(courses), "courses": simplified})

@app.route("/faqs", methods=["GET"])
def get_faqs():
    faqs = sheets_service.get_all_faqs()
    simplified = [{
        "id": f.get("faq_id"),
        "keywords": f.get("q_keywords"),
        "answer_preview": (f.get("answer", "")[:100] + "...") if f.get("answer") else ""
    } for f in faqs]

    return jsonify({"count": len(faqs), "faqs": simplified})

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint олдсонгүй"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Server алдаа: {error}")
    return jsonify({"error": "Дотоод серверийн алдаа"}), 500

# ======================
# Үндсэн
# ======================
if __name__ == "__main__":
    required_envs = ["SHEET_ID", "OPENAI_API_KEY", "GOOGLE_CREDENTIALS_JSON"]
    missing = [env for env in required_envs if not os.getenv(env)]

    if missing:
        logger.warning(f"⚠️ Дараах environment variable дутуу байна: {missing}")
        logger.warning("Үйлчилгээ дутуу тохиргоотойгоор эхэлж магадгүй...")

    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"🚀 Way Academy Chatbot Server {port} порт дээр эхэллээ...")
    logger.info(f"📊 Google Sheets ID: {app.config['SHEET_ID']}")
    logger.info(f"🤖 OpenAI Model: {app.config['OPENAI_MODEL']}")

    app.run(host="0.0.0.0", port=port, debug=debug_mode)
