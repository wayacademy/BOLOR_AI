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
import google.generativeai as genai
from dotenv import load_dotenv

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "{}")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")

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
    if not pipe_str:
        return []
    return [normalize_text(x) for x in str(pipe_str).split("|") if normalize_text(x)]

def safe_float(x, default=999.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

PRICE_INTENT_PATTERNS = [
    r"\bүнэ\b", r"\bтөлбөр\b", r"\bхөнгөлөлт\b", r"\bearly\b", r"\bbird\b",
    r"\bбүртг(үүлэх|үүлнэ|элт)\b", r"\bapply\b", r"\bapplication\b",
    r"\bтөлөх\b", r"\bхувааж\b", r"\bpocketzero\b"
]
def is_price_or_payment_question(msg: str) -> bool:
    m = normalize_text(msg)
    return any(re.search(p, m) for p in PRICE_INTENT_PATTERNS)

# course mention / hint detection (өөрчлөх боломжтой)
COURSE_HINT_PATTERNS = [
    r"\bsdm\b", r"\bda\b", r"\bitba\b", r"\bpz\b",
    r"strategic digital marketing", r"data analyst", r"it business analyst", r"project zero",
    r"маркетинг", r"аналист", r"бизнес аналист", r"agent", r"чатбот", r"n8n"
]
def has_course_hint(msg: str) -> bool:
    m = normalize_text(msg)
    return any(re.search(p, m) for p in COURSE_HINT_PATTERNS)

# Attachment URL (FB/IG CDN, lookaside гэх мэт) шалгах
def looks_like_attachment_url(msg: str) -> bool:
    m = normalize_text(msg)
    if not m:
        return False
    return (
        m.startswith("http://")
        or m.startswith("https://")
    )

# ======================
# Google Sheets Service
# ======================
class GoogleSheetsService:
    FAQ_TAB = "FAQ_BOLOR"
    COURSES_TAB = "COURSE_BOLOR"
    RULES_TAB = "CHATBOT_INTERNAL_RULES"  # sheet дээрх бодит нэр

    def __init__(self):
        self.sheet_id = app.config.get("SHEET_ID")
        self.service = None
        self._cache: Dict[str, Tuple[datetime, List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        self._initialize_service()

    def _initialize_service(self):
        try:
            if not self.sheet_id:
                logger.warning("⚠️ SHEET_ID тохируулагдаагүй байна.")
                return

            creds_str = app.config.get("CREDENTIALS_JSON", "{}")
            if not creds_str or creds_str == "{}":
                logger.warning("⚠️ GOOGLE_CREDENTIALS_JSON хоосон байна.")
                return

            credentials_info = json.loads(creds_str)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
            )
            self.service = build("sheets", "v4", credentials=credentials)
            logger.info("✅ Google Sheets API эхэллээ")
        except Exception as e:
            logger.error(f"❌ Sheets init error: {e}", exc_info=True)

    def _is_cache_valid(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            expires_at, _ = self._cache[key]
            return datetime.utcnow() < expires_at

    def _set_cache(self, key: str, data: List[Dict[str, Any]]):
        expires_at = datetime.utcnow() + timedelta(seconds=int(app.config.get("CACHE_TTL", 300)))
        with self._lock:
            self._cache[key] = (expires_at, data)

    def _get_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            if key not in self._cache:
                return None
            return self._cache[key][1]

    def _rows_to_dicts(self, values: List[List[str]]) -> List[Dict[str, Any]]:
        if not values:
            return []

        raw_headers = values[0]
        headers: List[str] = []
        for i, h in enumerate(raw_headers):
            h2 = str(h).strip()
            headers.append(h2 if h2 else f"col_{i}")

        data: List[Dict[str, Any]] = []
        for row in values[1:]:
            item: Dict[str, Any] = {}
            for i, header in enumerate(headers):
                item[header] = row[i] if i < len(row) else ""

            is_active = str(item.get("is_active", "True")).strip().lower()
            if is_active in ["true", "yes", "1", "active", ""]:
                if "payment_options" in item and "|" in str(item["payment_options"]):
                    item["payment_options"] = str(item["payment_options"]).replace("|", "\n")
                data.append(item)

        return data

    def refresh_all(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.service:
            logger.error("❌ Sheets service initialized биш байна.")
            return {"faq": [], "courses": [], "rules": []}

        try:
            sheet = self.service.spreadsheets()
            ranges = [
                f"{self.FAQ_TAB}!A:Z",
                f"{self.COURSES_TAB}!A:Z",
                f"{self.RULES_TAB}!A:Z",
            ]
            result = sheet.values().batchGet(spreadsheetId=self.sheet_id, ranges=ranges).execute()
            value_ranges = result.get("valueRanges", [])

            faq_values = value_ranges[0].get("values", []) if len(value_ranges) > 0 else []
            course_values = value_ranges[1].get("values", []) if len(value_ranges) > 1 else []
            rules_values = value_ranges[2].get("values", []) if len(value_ranges) > 2 else []

            faq_data = self._rows_to_dicts(faq_values)
            course_data = self._rows_to_dicts(course_values)
            rules_data = self._rows_to_dicts(rules_values)

            self._set_cache("faq_cache", faq_data)
            self._set_cache("courses_cache", course_data)
            self._set_cache("rules_cache", rules_data)

            logger.info(f"✅ Data refreshed: FAQ={len(faq_data)}, Courses={len(course_data)}, Rules={len(rules_data)}")
            return {"faq": faq_data, "courses": course_data, "rules": rules_data}
        except Exception as e:
            logger.error(f"❌ BatchGet error: {e}", exc_info=True)
            return {"faq": [], "courses": [], "rules": []}

    def get_all_courses(self) -> List[Dict[str, Any]]:
        if not self._is_cache_valid("courses_cache"):
            self.refresh_all()
        courses = self._get_cache("courses_cache") or []
        return sorted(courses, key=lambda x: safe_float(x.get("priority", 999) or 999, 999.0))

    def get_all_faqs(self) -> List[Dict[str, Any]]:
        if not self._is_cache_valid("faq_cache"):
            self.refresh_all()
        faqs = self._get_cache("faq_cache") or []
        return sorted(faqs, key=lambda x: safe_float(x.get("priority", 999) or 999, 999.0))

    # -------- Router --------
    def match_best_faq(self, user_message: str) -> Optional[Dict[str, Any]]:
        msg = normalize_text(user_message)
        if not msg:
            return None

        faqs = self.get_all_faqs()
        best = None
        best_score = 0
        best_priority = 999.0

        for f in faqs:
            kws = split_keywords(f.get("q_keywords", ""))
            if not kws:
                continue

            score = 0
            for kw in kws:
                if kw and kw in msg:
                    score += len(kw)

            if score <= 0:
                continue

            pr = safe_float(f.get("priority", 999) or 999, 999.0)
            if score > best_score or (score == best_score and pr < best_priority):
                best = f
                best_score = score
                best_priority = pr

        return best

    def match_best_course(self, user_message: str) -> Optional[Dict[str, Any]]:
        msg = normalize_text(user_message)
        if not msg:
            return None

        courses = self.get_all_courses()
        best = None
        best_score = 0
        best_priority = 999.0

        for c in courses:
            kws = split_keywords(c.get("keywords", ""))
            score = 0
            for kw in kws:
                if kw and kw in msg:
                    score += len(kw)

            cname = normalize_text(c.get("course_name", ""))
            if cname and cname in msg:
                score += 10

            if score <= 0:
                continue

            pr = safe_float(c.get("priority", 999) or 999, 999.0)
            if score > best_score or (score == best_score and pr < best_priority):
                best = c
                best_score = score
                best_priority = pr

        return best

# ======================
# Gemini Service (FACT-only rewrite)
# ======================
class GeminiService:
    def __init__(self):
        self.api_key = app.config.get("GEMINI_API_KEY")
        self.primary_model_name = app.config.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.fallback_model_name = app.config.get("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")
        self.model = None
        self.model_name = None
        self._initialize_client()

    def _initialize_client(self):
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY олдсонгүй.")
            return

        genai.configure(api_key=self.api_key)

        system_instruction = """Та бол Way Academy-гийн албан ёсны туслах.
ХАТУУ ДҮРЭМ:
- Зөвхөн FACTS-д байгаа мэдээллээр хариул. Шинэ мэдээлэл зохиож нэмэхийг хориглоно.
- Нэг хариултад нэг intent.
- Хариулт 120–180 үг орчим (хэт урт биш).
- Хариултын эхэнд тогтмол мэндчилгээ бүү давт.
- Хэрэв FACTS-д байхгүй зүйл асуувал “Энэ талаар мэдээлэл алга байна” гэж хэлээд холбоо барих сувгийг өг.
"""

        try:
            self.model = genai.GenerativeModel(
                model_name=self.primary_model_name,
                system_instruction=system_instruction
            )
            self.model_name = self.primary_model_name
            logger.info(f"✅ Gemini model: {self.model_name}")
            return
        except Exception as e:
            logger.warning(f"⚠️ Primary model init failed ({self.primary_model_name}): {e}")

        try:
            self.model = genai.GenerativeModel(
                model_name=self.fallback_model_name,
                system_instruction=system_instruction
            )
            self.model_name = self.fallback_model_name
            logger.info(f"✅ Gemini fallback model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Fallback model init failed ({self.fallback_model_name}): {e}", exc_info=True)

    def rewrite_from_facts(self, user_question: str, intent: str, facts: Dict[str, Any], force_escalation: bool) -> str:
        contact = "Холбогдох: 91117577, 99201187 | hello@wayconsulting.io"
        location = "Байршил: Galaxy Tower, 7 давхар, 705"

        if not self.model:
            return self._fallback(intent, facts, force_escalation, contact, location)

        facts_json = json.dumps(facts, ensure_ascii=False)

        escalation_line = ""
        if force_escalation:
            escalation_line = "Яг тохирох нөхцөл, суудал баталгаажуулахын тулд 91117577, 99201187 эсвэл hello@wayconsulting.io-р холбогдоорой."

        prompt = f"""
INTENT: {intent}

USER QUESTION:
{user_question}

FACTS (JSON):
{facts_json}

OUTPUT FORMAT (заавал):
1) 1 өгүүлбэр: шууд хариулт
2) 2–4 bullet: зөвхөн FACTS-д байгаа баримтууд
3) 1 өгүүлбэр CTA: дараагийн алхам
- Хэрэв escalation шаардлагатай бол CTA хэсэгт доорх өгүүлбэрийг заавал оруул:
"{escalation_line}"
- Төгсгөлд холбоо барихыг нэг мөрөөр өг:
"{contact} | {location}"
"""
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.35,
                    max_output_tokens=340
                )
            )
            text = (resp.text or "").strip()
            if not text:
                return self._fallback(intent, facts, force_escalation, contact, location)
            return text
        except Exception as e:
            logger.error(f"❌ rewrite_from_facts error: {e}", exc_info=True)
            return self._fallback(intent, facts, force_escalation, contact, location)

    def _fallback(self, intent: str, facts: Dict[str, Any], force_escalation: bool, contact: str, location: str) -> str:
        bullets = []
        for k in ["course_name", "duration", "schedule_1", "schedule_2", "teacher", "price_full", "price_discount", "payment_options"]:
            v = facts.get(k)
            if isinstance(v, str):
                v = v.strip()
            if v:
                bullets.append(f"• {k}: {v}")
            if len(bullets) >= 4:
                break

        main = "Энэ талаар мэдээлэл алга байна."
        if intent == "FAQ" and facts.get("answer"):
            main = str(facts.get("answer")).strip()
        if intent == "COURSE" and facts.get("course_name"):
            main = str(facts.get("course_name")).strip()

        cta = "Дэлгэрэнгүй мэдээлэл авах бол холбогдоорой."
        if force_escalation:
            cta = "Яг тохирох нөхцөл, суудал баталгаажуулахын тулд 91117577, 99201187 эсвэл hello@wayconsulting.io-р холбогдоорой."

        return f"{main}\n" + ("\n".join(bullets) + "\n" if bullets else "") + f"{cta}\n{contact} | {location}"

# ======================
# Instances
# ======================
sheets_service = GoogleSheetsService()
ai_service = GeminiService()

# ======================
# Response helpers
# ======================
def programs_list_fact(courses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    items = []
    for c in courses:
        cid = (c.get("course_id") or "").strip()
        nm = (c.get("course_name") or "").strip()
        if nm:
            items.append({"course_id": cid, "course_name": nm})
    return items

def clarify_for_course() -> str:
    # Товчлол ашиглахгүйгээр ойлгомжтойгоор асууж байна (user request)
    return "Та аль хөтөлбөрийн талаар асууж байна вэ? (Стратегийн Дижитал Маркетер / Data Analyst / IT Business Analyst / Project Zero)"

def ask_text_instructions() -> str:
    # Жишээ дээр товчлол ашиглахгүй
    return "Та асуултаа текстээр бичээд явуулна уу. (Ж: “Стратегийн Дижитал Маркетер хөтөлбөрийн үнэ хэд вэ?”)"

# ======================
# Routes
# ======================
@app.route("/")
def index():
    return jsonify({"status": "active", "gemini_model": ai_service.model_name})

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "services": {
            "sheets": bool(sheets_service.service),
            "gemini": bool(ai_service.model),
            "gemini_model": ai_service.model_name
        }
    })

@app.route("/manychat/webhook", methods=["POST"])
def manychat_webhook():
    """
    ManyChat External Request Best Practice:
    - External Request never displays response automatically.
    - We must return simple JSON for Response Mapping, e.g. {"reply": "..."}.
    """
    try:
        if not request.is_json:
            return jsonify({}), 400

        data = request.get_json() or {}
        subscriber_id = str(data.get("subscriber_id", "")).strip()
        user_message = (data.get("message") or "").strip()

        if not subscriber_id:
            return jsonify({}), 400

        logger.info(f"📩 Webhook: {subscriber_id} -> {user_message}")

        # 1) Empty message guardrail
        if not user_message or not user_message.strip():
            return jsonify({"reply": ask_text_instructions()})

        # 2) Attachment URL guardrail
        if looks_like_attachment_url(user_message):
            return jsonify({"reply": ask_text_instructions()})

        # Load data (cached)
        faqs = sheets_service.get_all_faqs()
        courses = sheets_service.get_all_courses()

        # Clarify gate: price/payment asked but course not specified
        if is_price_or_payment_question(user_message) and not has_course_hint(user_message):
            return jsonify({"reply": clarify_for_course()})

        # Router: best FAQ or best COURSE
        best_faq = sheets_service.match_best_faq(user_message)
        best_course = sheets_service.match_best_course(user_message)

        chosen_intent = None
        chosen_obj = None

        if best_faq and best_course:
            faq_pr = safe_float(best_faq.get("priority", 999) or 999, 999.0)
            crs_pr = safe_float(best_course.get("priority", 999) or 999, 999.0)
            chosen_intent, chosen_obj = ("FAQ", best_faq) if faq_pr <= crs_pr else ("COURSE", best_course)
        elif best_faq:
            chosen_intent, chosen_obj = "FAQ", best_faq
        elif best_course:
            chosen_intent, chosen_obj = "COURSE", best_course
        else:
            chosen_intent, chosen_obj = "FALLBACK", None

        # Escalation gate: price/payment only when we actually respond with info
        force_escalation = is_price_or_payment_question(user_message) and chosen_intent in ["FAQ", "COURSE"]

        # Build facts
        if chosen_intent == "FAQ":
            facts = {
                "faq_id": chosen_obj.get("faq_id"),
                "q_keywords": chosen_obj.get("q_keywords"),
                "answer": chosen_obj.get("answer"),
                "contact_phone": "91117577, 99201187",
                "contact_email": "hello@wayconsulting.io",
                "location": "Galaxy Tower, 7 давхар, 705"
            }
            if normalize_text(chosen_obj.get("faq_id", "")) == "faq_programs":
                facts["programs"] = programs_list_fact(courses)

            text = ai_service.rewrite_from_facts(
                user_question=user_message,
                intent="FAQ",
                facts=facts,
                force_escalation=force_escalation
            )

        elif chosen_intent == "COURSE":
            c = chosen_obj
            facts = {
                "course_id": c.get("course_id"),
                "course_name": c.get("course_name"),
                "description": c.get("description"),
                "teacher": c.get("teacher"),
                "duration": c.get("duration"),
                "schedule_1": c.get("schedule_1"),
                "schedule_2": c.get("schedule_2"),
                "price_full": c.get("price_full"),
                "price_discount": c.get("price_discount"),
                "price_discount_until": c.get("price_discount_until"),
                "payment_options": c.get("payment_options"),
                "application_link": c.get("application_link"),
                "cta_caption": c.get("cta_caption"),
                "contact_phone": "91117577, 99201187",
                "contact_email": "hello@wayconsulting.io",
                "location": "Galaxy Tower, 7 давхар, 705"
            }

            text = ai_service.rewrite_from_facts(
                user_question=user_message,
                intent="COURSE",
                facts=facts,
                force_escalation=force_escalation
            )

        else:
            facts = {
                "programs": programs_list_fact(courses),
                "contact_phone": "91117577, 99201187",
                "contact_email": "hello@wayconsulting.io",
                "location": "Galaxy Tower, 7 давхар, 705"
            }
            text = ai_service.rewrite_from_facts(
                user_question=user_message,
                intent="FALLBACK",
                facts=facts,
                force_escalation=False
            )

        # IMPORTANT: Return only {"reply": "..."} for ManyChat External Request
        return jsonify({"reply": text})

    except Exception as e:
        logger.error(f"❌ Webhook Error: {e}", exc_info=True)
        return jsonify({"reply": "Уучлаарай, системд алдаа гарлаа. Дахин оролдоно уу."}), 200

@app.route("/test", methods=["POST"])
def test_endpoint():
    data = request.json or {}
    q = (data.get("question") or "").strip() or "Стратегийн Дижитал Маркетер хөтөлбөрийн үнэ хэд вэ?"

    courses = sheets_service.get_all_courses()

    if not q:
        return jsonify({"question": q, "route": "EMPTY", "response": ask_text_instructions()})

    if looks_like_attachment_url(q):
        return jsonify({"question": q, "route": "ATTACHMENT", "response": ask_text_instructions()})

    if is_price_or_payment_question(q) and not has_course_hint(q):
        return jsonify({"question": q, "route": "CLARIFY", "response": clarify_for_course()})

    best_faq = sheets_service.match_best_faq(q)
    best_course = sheets_service.match_best_course(q)

    chosen_intent = "FALLBACK"
    chosen_obj = None
    if best_faq and best_course:
        faq_pr = safe_float(best_faq.get("priority", 999) or 999, 999.0)
        crs_pr = safe_float(best_course.get("priority", 999) or 999, 999.0)
        chosen_intent, chosen_obj = ("FAQ", best_faq) if faq_pr <= crs_pr else ("COURSE", best_course)
    elif best_faq:
        chosen_intent, chosen_obj = "FAQ", best_faq
    elif best_course:
        chosen_intent, chosen_obj = "COURSE", best_course

    force_escalation = is_price_or_payment_question(q) and chosen_intent in ["FAQ", "COURSE"]

    if chosen_intent == "FAQ":
        facts = {"faq_id": chosen_obj.get("faq_id"), "answer": chosen_obj.get("answer")}
        if normalize_text(chosen_obj.get("faq_id", "")) == "faq_programs":
            facts["programs"] = programs_list_fact(courses)
        resp = ai_service.rewrite_from_facts(q, "FAQ", facts, force_escalation)
    elif chosen_intent == "COURSE":
        c = chosen_obj
        facts = {
            "course_id": c.get("course_id"),
            "course_name": c.get("course_name"),
            "duration": c.get("duration"),
            "schedule_1": c.get("schedule_1"),
            "schedule_2": c.get("schedule_2"),
            "teacher": c.get("teacher"),
            "price_full": c.get("price_full"),
            "price_discount": c.get("price_discount"),
            "payment_options": c.get("payment_options"),
            "application_link": c.get("application_link"),
        }
        resp = ai_service.rewrite_from_facts(q, "COURSE", facts, force_escalation)
    else:
        facts = {"programs": programs_list_fact(courses)}
        resp = ai_service.rewrite_from_facts(q, "FALLBACK", facts, False)

    return jsonify({"question": q, "route": chosen_intent, "response": resp, "gemini_model": ai_service.model_name})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
