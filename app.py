import os
import json
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError

# ======================
# Logging Configuration
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# ======================
# Rate Limiter
# ======================
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ======================
# Configuration
# ======================
class Config:
    SHEET_ID = os.getenv("SHEET_ID")
    GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "{}")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
    ENABLE_PARALLEL_MATCHING = os.getenv("ENABLE_PARALLEL_MATCHING", "true").lower() == "true"
    
    # Intent thresholds
    FAQ_MIN_MATCH_SCORE = float(os.getenv("FAQ_MIN_MATCH_SCORE", "0.3"))
    COURSE_MIN_MATCH_SCORE = float(os.getenv("COURSE_MIN_MATCH_SCORE", "0.4"))
    
    # Response templates
    FALLBACK_RESPONSE = os.getenv("FALLBACK_RESPONSE", 
                                   "Уучлаарай, би таны асуултанд хариулж чадахгүй байна. "
                                   "Дэлгэрэнгүй мэдээлэл авахыг хүсвэл 91117577 дугаарт холбогдоно уу.")
    ERROR_RESPONSE = os.getenv("ERROR_RESPONSE",
                               "Системийн алдаа гарлаа. Та түр хүлээгээд дахин оролдоно уу "
                               "эсвэл 91117577-р холбогдоорой.")

app.config.from_object(Config)

# ======================
# Helper Functions
# ======================
def normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    if not s:
        return ""
    
    # Remove extra whitespace
    s = s.strip()
    
    # Convert to lowercase for Mongolian and English
    s = s.lower()
    
    # Remove multiple spaces
    s = re.sub(r"\s+", " ", s)
    
    # Remove punctuation (optional, can be adjusted)
    s = re.sub(r"[^\w\s\u0400-\u04FF\u1800-\u18AF]", "", s)
    
    return s

def split_keywords(pipe_str: str) -> List[str]:
    """Split keywords and normalize them."""
    if not pipe_str:
        return []
    
    keywords = []
    for kw in str(pipe_str).split("|"):
        normalized = normalize_text(kw)
        if normalized and len(normalized) > 1:  # Filter out very short keywords
            keywords.append(normalized)
    return keywords

def calculate_match_score(message: str, keywords: List[str]) -> float:
    """Calculate match score between message and keywords."""
    if not keywords:
        return 0.0
    
    message_words = set(normalize_text(message).split())
    matches = 0
    
    for keyword in keywords:
        # Check for exact match or word boundary match
        if len(keyword.split()) > 1:
            # Multi-word keyword
            if keyword in normalize_text(message):
                matches += len(keyword.split())
        else:
            # Single word keyword
            if re.search(rf"\b{re.escape(keyword)}\b", normalize_text(message)):
                matches += 1
    
    # Normalize score by keyword count
    return matches / len(keywords) if keywords else 0.0

def is_price_question(msg: str) -> bool:
    """Check if question is about pricing."""
    price_patterns = [
        r"\bүнэ\b", r"\bтөлбөр\b", r"\bхөнгөлөлт\b", r"\b₮\b", r"\bтөгрөг\b",
        r"\bдоллар\b", r"\b\\$\b", r"\bprice\b", r"\bfee\b", r"\bcost\b",
        r"\bapply\b", r"\bpayment\b", r"\bearly\b", r"\bхугацаа\b",
        r"\bхэд\b.*\bүнэ\b", r"\bhow much\b", r"\bхямдрал\b", r"\bдискаунт\b"
    ]
    
    m = normalize_text(msg)
    return any(re.search(p, m) for p in price_patterns)

def is_urgent_question(msg: str) -> bool:
    """Check if question is urgent."""
    urgent_patterns = [
        r"\bяаралтай\b", r"\bтүргэн\b", r"\bнохой\b.*\bхазуул\b", r"\bemergency\b",
        r"\burgent\b", r"\bаюултай\b", r"\bосол\b", r"\bнөхөр\b.*\bцохи\b"
    ]
    
    m = normalize_text(msg)
    return any(re.search(p, m) for p in urgent_patterns)

def has_url(msg: str) -> bool:
    """Check if message contains URL."""
    url_patterns = [
        r"https?://\S+",
        r"www\.\S+",
        r"\S+\.(com|org|net|edu|gov|io|mn)\b"
    ]
    return any(re.search(p, msg) for p in url_patterns)

def sanitize_input(text: str) -> str:
    """Sanitize user input."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long
    max_len = app.config['MAX_QUESTION_LENGTH']
    if len(text) > max_len:
        text = text[:max_len]
    
    # Remove potentially harmful patterns
    text = re.sub(r'[<>\"\']', '', text)
    
    return text

# ======================
# Google Sheets Service
# ======================
class GoogleSheetsService:
    FAQ_TAB = "FAQ_BOLOR"
    COURSE_TAB = "COURSE_BOLOR"
    
    def __init__(self):
        self.sheet_id = app.config["SHEET_ID"]
        self._cache = {}
        self._cache_timestamps = {}
        self._lock = threading.Lock()
        self.service = None
        self._init_service()
    
    def _init_service(self):
        """Initialize Google Sheets service with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                creds = json.loads(app.config["GOOGLE_CREDENTIALS_JSON"])
                credentials = service_account.Credentials.from_service_account_info(
                    creds,
                    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
                )
                self.service = build("sheets", "v4", credentials=credentials, cache_discovery=False)
                logger.info("✅ Google Sheets service initialized successfully")
                return
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse Google credentials JSON: {e}")
                break
            except Exception as e:
                logger.error(f"❌ Google Sheets init failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid."""
        if key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.utcnow() - self._cache_timestamps[key]
        return cache_age.total_seconds() < app.config["CACHE_TTL"]
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache if valid."""
        with self._lock:
            if key in self._cache and self._is_cache_valid(key):
                return self._cache[key]
            return None
    
    def _cache_set(self, key: str, data: Any):
        """Set item in cache with timestamp."""
        with self._lock:
            self._cache[key] = data
            self._cache_timestamps[key] = datetime.utcnow()
    
    def _rows_to_dicts(self, values: List[List]) -> List[Dict]:
        """Convert sheet rows to list of dictionaries."""
        if not values or len(values) < 2:
            return []
        
        headers = [str(h).strip() for h in values[0]]
        out = []
        
        for row in values[1:]:
            item = {}
            for i, h in enumerate(headers):
                if i < len(row):
                    item[h] = str(row[i]).strip() if row[i] is not None else ""
                else:
                    item[h] = ""
            
            # Filter inactive items
            is_active = str(item.get("is_active", "true")).lower()
            if is_active in ["true", "1", "yes", "y", ""]:
                out.append(item)
        
        return out
    
    def refresh(self) -> Tuple[List[Dict], List[Dict]]:
        """Refresh data from Google Sheets with error handling."""
        if not self.service:
            logger.warning("Google Sheets service not initialized")
            return [], []
        
        try:
            sheet = self.service.spreadsheets()
            res = sheet.values().batchGet(
                spreadsheetId=self.sheet_id,
                ranges=[f"{self.FAQ_TAB}!A:Z", f"{self.COURSE_TAB}!A:Z"],
                valueRenderOption="FORMATTED_VALUE"
            ).execute()
            
            faq_values = res["valueRanges"][0].get("values", [])
            course_values = res["valueRanges"][1].get("values", [])
            
            faq_data = self._rows_to_dicts(faq_values)
            course_data = self._rows_to_dicts(course_values)
            
            self._cache_set("faq", faq_data)
            self._cache_set("course", course_data)
            
            logger.info(f"✅ Data refreshed: {len(faq_data)} FAQs, {len(course_data)} courses")
            return faq_data, course_data
            
        except HttpError as e:
            logger.error(f"❌ Google Sheets API error: {e}")
            return [], []
        except Exception as e:
            logger.error(f"❌ Error refreshing data: {e}")
            return [], []
    
    def get_faqs(self) -> List[Dict]:
        """Get FAQs from cache or refresh."""
        data = self._cache_get("faq")
        if data is not None:
            return data
        
        # Try to refresh
        try:
            faq_data, _ = self.refresh()
            return faq_data
        except Exception:
            return []
    
    def get_courses(self) -> List[Dict]:
        """Get courses from cache or refresh."""
        data = self._cache_get("course")
        if data is not None:
            return data
        
        # Try to refresh
        try:
            _, course_data = self.refresh()
            return course_data
        except Exception:
            return []
    
    def match_faq(self, msg: str) -> Optional[Dict]:
        """Find best matching FAQ for the message."""
        msg_normalized = normalize_text(msg)
        best_match = None
        best_score = 0
        
        for faq in self.get_faqs():
            keywords = split_keywords(faq.get("q_keywords", ""))
            score = calculate_match_score(msg_normalized, keywords)
            
            if score > best_score and score >= app.config["FAQ_MIN_MATCH_SCORE"]:
                best_score = score
                best_match = faq
        
        if best_match:
            logger.info(f"FAQ match found: {best_match.get('question', 'N/A')} (score: {best_score:.2f})")
        
        return best_match
    
    def match_course(self, msg: str) -> Optional[Dict]:
        """Find best matching course for the message."""
        msg_normalized = normalize_text(msg)
        best_match = None
        best_score = 0
        
        for course in self.get_courses():
            keywords = split_keywords(course.get("keywords", ""))
            score = calculate_match_score(msg_normalized, keywords)
            
            if score > best_score and score >= app.config["COURSE_MIN_MATCH_SCORE"]:
                best_score = score
                best_match = course
        
        if best_match:
            logger.info(f"Course match found: {best_match.get('name', 'N/A')} (score: {best_score:.2f})")
        
        return best_match

# ======================
# OpenAI Service
# ======================
class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=app.config["OPENAI_API_KEY"])
        self.model = app.config["OPENAI_MODEL"]
        self.max_retries = 3
    
    def generate_response(self, question: str, intent: str, facts: Dict, 
                         escalate: bool = False, is_urgent: bool = False) -> str:
        """Generate response using OpenAI with retry logic."""
        
        system_prompt = """Та бол Way Academy-гийн албан ёсны AI зөвлөх.
ХАТУУ ДҮРЭМ:
1. ЗӨВХӨН FACTS JSON-с мэдээлэл ашиглах
2. Шинэ зүйл зохиох, таамаглахыг хориглоно
3. Нэг intent дээр төвлөр
4. Хариулт нь 120–180 үг (2-4 өгүүлбэр)
5. Байгаа мэдээллээс заавал хариулах
6. Хэрэв мэдээлэл дутуу байвал хүнтэй холбоотой байхыг зөвлөх

Формат:
- 1 өгүүлбэр хариулт
- 2-4 bullet цэг
- Дуусахад CTA (үйлдэл хийх уриалга)"""
        
        user_prompt = f"""
INTENT: {intent}
QUESTION: {question}

AVAILABLE FACTS (use ONLY these):
{json.dumps(facts, ensure_ascii=False, indent=2)}

Хэрэглэгчийн асуултанд ДЭЭРХ FACTS-с хариулна уу.
Хэрэв facts дотор хариулт байхгүй бол хүнтэй холбоотой байхыг зөвлөнө.
"""
        
        if is_urgent:
            user_prompt += "\nАНХААРУУЛГА: Энэ асуулт яаралтай бөгөөд шууд хүнтэй холбоотой байхыг зөвлөх ёстой."
        
        if escalate:
            user_prompt += "\nCTA: Дэлгэрэнгүй мэдээлэл авах бол 91117577 дугаарт холбоо барина уу эсвэл hello@wayconsulting.io"
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.35,
                    max_tokens=350,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    timeout=30
                )
                
                reply = response.choices[0].message.content.strip()
                
                # Validate response isn't too short
                if len(reply) < 20:
                    raise ValueError("Response too short")
                
                logger.info(f"OpenAI response generated (attempt {attempt + 1})")
                return reply
                
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Rate limit hit, retrying... ({attempt + 1}/{self.max_retries})")
                import time
                time.sleep(2 ** attempt)
                
            except (APIConnectionError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Connection error, retrying... ({attempt + 1}/{self.max_retries})")
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                raise
        
        # Fallback if all retries fail
        return app.config["ERROR_RESPONSE"]

# ======================
# Application Instances
# ======================
sheets = GoogleSheetsService()
ai = OpenAIService()
executor = ThreadPoolExecutor(max_workers=2) if app.config["ENABLE_PARALLEL_MATCHING"] else None

# ======================
# ManyChat Webhook Endpoint
# ======================
@app.route("/manychat/webhook", methods=["POST"])
@limiter.limit("10 per minute")
def webhook():
    """Handle ManyChat webhook requests."""
    start_time = datetime.utcnow()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"reply": "Invalid JSON format"}), 400
        
        data = request.get_json(silent=True) or {}
        msg = data.get("message", "").strip()
        user_id = data.get("user_id", "unknown")
        
        logger.info(f"Request from user {user_id}: {msg[:100]}...")
        
        # Sanitize and validate input
        msg = sanitize_input(msg)
        
        if not msg:
            return jsonify({"reply": "Хоосон асуулт илгээсэн байна. Асуултаа бичнэ үү."}), 200
        
        if has_url(msg):
            return jsonify({"reply": "Асуултаа текстээр бичнэ үү. Холбоос илгээх боломжгүй."}), 200
        
        if len(msg) < 2:
            return jsonify({"reply": "Асуулт маш богино байна. Дэлгэрэнгүй бичнэ үү."}), 200
        
        # Check for urgent questions
        urgent = is_urgent_question(msg)
        if urgent:
            logger.warning(f"Urgent question detected from user {user_id}: {msg}")
        
        # Parallel matching if enabled
        if executor and app.config["ENABLE_PARALLEL_MATCHING"]:
            future_faq = executor.submit(sheets.match_faq, msg)
            future_course = executor.submit(sheets.match_course, msg)
            
            faq = future_faq.result(timeout=5)
            course = future_course.result(timeout=5)
        else:
            faq = sheets.match_faq(msg)
            course = sheets.match_course(msg)
        
        # Determine intent and facts
        intent = "FALLBACK"
        facts = {}
        
        if course:
            intent = "COURSE"
            facts = course
        elif faq:
            intent = "FAQ"
            facts = {"answer": faq.get("answer")}
        
        # Check for price questions without course match
        if is_price_question(msg) and not course:
            return jsonify({
                "reply": "Ямар хөтөлбөрийн үнэ сонирхож байна вэ? Жишээ нь: 'IELTS', 'GMAT', 'Орчуулагч' гэх мэт."
            }), 200
        
        # Generate response
        if intent == "FALLBACK":
            reply = app.config["FALLBACK_RESPONSE"]
        else:
            try:
                reply = ai.generate_response(
                    question=msg,
                    intent=intent,
                    facts=facts,
                    escalate=is_price_question(msg) or urgent,
                    is_urgent=urgent
                )
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                reply = app.config["ERROR_RESPONSE"]
        
        # Log response time
        response_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Response generated in {response_time:.2f}s for user {user_id}")
        
        return jsonify({"reply": reply}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({
            "reply": app.config["ERROR_RESPONSE"]
        }), 200

# ======================
# Admin Endpoints
# ======================
@app.route("/admin/refresh", methods=["POST"])
def admin_refresh():
    """Admin endpoint to manually refresh cache."""
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {os.getenv('ADMIN_TOKEN', 'default_token')}":
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        faq_count, course_count = sheets.refresh()
        return jsonify({
            "status": "success",
            "faq_count": len(faq_count),
            "course_count": len(course_count),
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/cache/stats", methods=["GET"])
def cache_stats():
    """Get cache statistics."""
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {os.getenv('ADMIN_TOKEN', 'default_token')}":
        return jsonify({"error": "Unauthorized"}), 401
    
    stats = {
        "faq_count": len(sheets.get_faqs()),
        "course_count": len(sheets.get_courses()),
        "cache_timestamps": sheets._cache_timestamps,
        "cache_ttl": app.config["CACHE_TTL"]
    }
    return jsonify(stats), 200

# ======================
# Health Check Endpoint
# ======================
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    health_status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "openai": bool(app.config["OPENAI_API_KEY"]),
            "google_sheets": bool(sheets.service),
            "cache_initialized": bool(sheets._cache)
        },
        "cache": {
            "faq_count": len(sheets.get_faqs()),
            "course_count": len(sheets.get_courses())
        }
    }
    
    # Check OpenAI connectivity
    if app.config["OPENAI_API_KEY"]:
        try:
            test_client = OpenAI(api_key=app.config["OPENAI_API_KEY"])
            test_client.models.list(timeout=5)
            health_status["services"]["openai_connectivity"] = "ok"
        except Exception as e:
            health_status["services"]["openai_connectivity"] = f"error: {str(e)}"
    
    # Check Google Sheets connectivity
    if sheets.service:
        try:
            sheet = sheets.service.spreadsheets()
            sheet.get(spreadsheetId=sheets.sheet_id, fields="properties.title").execute()
            health_status["services"]["sheets_connectivity"] = "ok"
        except Exception as e:
            health_status["services"]["sheets_connectivity"] = f"error: {str(e)}"
    
    return jsonify(health_status), 200

# ======================
# Application Startup
# ======================
@app.before_first_request
def initialize_app():
    """Initialize application on first request."""
    logger.info("Initializing application...")
    
    # Refresh data on startup
    try:
        sheets.refresh()
        logger.info("✅ Initial data loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load initial data: {e}")

# ======================
# Error Handlers
# ======================
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "reply": "Хэт олон хүсэлт илгээсэн байна. Түр хүлээгээд дахин оролдоно уу."
    }), 429

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

# ======================
# Application Entry Point
# ======================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Flask application on port {port} (debug: {debug})")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)