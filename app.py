import os
import json
import logging
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai
import requests
from dotenv import load_dotenv

# Лог тохируулах
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables ачаалах
load_dotenv()

app = Flask(__name__)
CORS(app)  # CORS зөвшөөрөх

# ======================
# Конфигураци
# ======================
class Config:
    # Google Sheets API тохиргоо
    SHEET_ID = os.getenv('SHEET_ID', '1HG2o-2oJtYwCWoGQpC3HhC_n6_scR-cPrMB47U9yc90')
    CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON', '{}')
    
    # OpenAI API (Groq-д хэрэглэх боломжтой)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # ManyChat тохиргоо
    MANYCHAT_TOKEN = os.getenv('MANYCHAT_TOKEN')
    
    # Кэш хугацаа (секундээр)
    CACHE_TTL = 300  # 5 минут
    
    # ManyChat API URL
    MANYCHAT_API_URL = "https://api.manychat.com/fb/sending/sendContent"

app.config.from_object(Config)

# ======================
# Google Sheets Service
# ======================
class GoogleSheetsService:
    def __init__(self):
        self.sheet_id = app.config['SHEET_ID']
        self.service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Google Sheets API сервис эхлүүлэх"""
        try:
            credentials_json = app.config['CREDENTIALS_JSON']
            
            # JSON string эсвэл dict байна уу шалгах
            if isinstance(credentials_json, str):
                if credentials_json.strip() == '{}':
                    logger.error("❌ Google Credentials хоосон байна")
                    raise ValueError("Google Credentials хоосон байна")
                credentials_info = json.loads(credentials_json)
            else:
                credentials_info = credentials_json
                
            if not credentials_info:
                logger.error("❌ Google Credentials хоосон байна")
                raise ValueError("Google Credentials хоосон байна")
                
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            self.service = build('sheets', 'v4', credentials=credentials)
            logger.info("✅ Google Sheets API сервис эхлэв")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON буруу формат: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Google Sheets API эхлүүлэхэд алдаа: {e}")
            raise
    
    @lru_cache(maxsize=1)
    def get_cached_data(self, sheet_name: str, cache_key: str = ""):
        """Кэшлэсэн өгөгдөл авах - cache_key нь зөвхөн LRU cache дэмжихэд"""
        return self.get_sheet_data(sheet_name)
    
    def get_sheet_data(self, sheet_name: str) -> List[Dict[str, Any]]:
        """Google Sheets-ээс өгөгдөл унших"""
        try:
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=self.sheet_id,
                range=f"{sheet_name}!A:Z"
            ).execute()
            
            values = result.get('values', [])
            if not values:
                return []
            
            # Эхний мөрийг баганы нэр болгох
            headers = values[0]
            data = []
            
            for row in values[1:]:
                # Мөр бүрийг баганатай нь хослуулах
                item = {}
                for i, header in enumerate(headers):
                    if i < len(row):
                        item[header] = row[i]
                    else:
                        item[header] = ""
                
                # Зөвхөн идэвхтэй мөрүүдийг нэмэх
                if item.get('is_active', 'True').strip().lower() == 'true':
                    data.append(item)
            
            logger.info(f"✅ {sheet_name} хуудаснаас {len(data)} мөр уншлаа")
            return data
            
        except Exception as e:
            logger.error(f"❌ {sheet_name} хуудсыг уншихад алдаа: {e}")
            return []
    
    def get_all_courses(self) -> List[Dict[str, Any]]:
        """Бүх сургалтуудыг авах"""
        courses = self.get_cached_data("courses", "courses_cache")
        
        # Priority эрэмбээр жагсаах
        return sorted(courses, key=lambda x: float(x.get('priority', 999)))
    
    def get_all_faqs(self) -> List[Dict[str, Any]]:
        """Бүх FAQ-уудыг авах"""
        return self.get_cached_data("faq", "faq_cache")
    
    def get_course_by_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Түлхүүр үгээр сургалт хайх"""
        if not keyword or keyword.strip() == '':
            return None
            
        keyword_lower = keyword.lower().strip()
        courses = self.get_all_courses()
        
        for course in courses:
            # Түлхүүр үгсээр хайх
            keywords = course.get('keywords', '')
            if keywords and '|' in keywords:
                course_keywords = [k.strip().lower() for k in keywords.split('|')]
                if any(kw in keyword_lower for kw in course_keywords):
                    return course
            
            # Нэрээр хайх
            course_name = course.get('course_name', '').lower()
            if keyword_lower in course_name:
                return course
            
            # ID-аар хайх
            course_id = course.get('course_id', '').lower()
            if keyword_lower == course_id.lower():
                return course
        
        return None

# ======================
# AI Service (OpenAI/Groq)
# ======================
class AIService:
    def __init__(self):
        self.api_key = app.config['OPENAI_API_KEY']
        self.model = app.config['OPENAI_MODEL']
        
        if not self.api_key:
            logger.warning("⚠️ OpenAI/Groq API Key олдсонгүй!")
        else:
            openai.api_key = self.api_key
    
    def generate_response(self, user_question: str, context_data: Dict[str, Any]) -> str:
        """AI ашиглан хариулт үүсгэх"""
        try:
            if not self.api_key:
                return "Уучлаарай, AI сервис түр ажиллахгүй байна. Та 91117577 дугаарт залгана уу."
            
            # Монгол хэл дээрх систем prompt
            system_prompt = """Та бол Way Academy-гийн албан ёсны туслах чатбот. 
Дараах дүрмийг баримтлаарай:
1. ЗӨВХӨН өгөгдсөн мэдээллээс хариулт өгөх
2. Монгол хэлээр, найрсаг, товч хариулт өгөх
3. Үнэ, цаг, багшийн мэдээллийг тодорхой харуулах
4. Хэрэв мэдээлэл олдохгүй бол "Уучлаарай, энэ асуултанд хариулж чадахгүй байна" гэж хэлэх
5. Сургалтын нэр, ID (SDM, DA гэх мэт) зөв хэрэглэх
6. Хариултын төгсгөлд "Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу" гэж нэмэх"""
            
            # Context-ыг форматлах
            context_str = self._format_context(context_data)
            
            # Хэрэглэгчийн асуулт
            user_prompt = f"""Хэрэглэгчийн асуулт: {user_question}

Доорх мэдээллээс хариулт өгнө үү:
{context_str}

Хариулт:"""
            
            # Groq эсвэл OpenAI API дуудах
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                ai_response = response.choices[0].message.content.strip()
                
                # Хариулт хоосон эсэхийг шалгах
                if not ai_response or ai_response == "":
                    return "Уучлаарай, хариулт үүсгэхэд алдаа гарлаа. Нэмэлт мэдээлэл авах бол 91117577 дугаарт залгана уу."
                
                return ai_response
                
            except Exception as api_error:
                logger.error(f"❌ AI API алдаа: {api_error}")
                # Fallback хариулт
                return self._generate_fallback_response(user_question, context_data)
            
        except Exception as e:
            logger.error(f"❌ AI хариулт үүсгэхэд алдаа: {e}")
            return "Уучлаарай, хариулт үүсгэхэд алдаа гарлаа. Та дахин оролдоно уу эсвэл 91117577 дугаарт залгана уу."
    
    def _format_context(self, data: Dict[str, Any]) -> str:
        """Өгөгдлийг AI-д ойлгомжтой форматлах"""
        context_parts = []
        
        # Сургалтын мэдээлэл
        if 'courses' in data and data['courses']:
            context_parts.append("=== СУРГАЛТУУД ===")
            for course in data['courses']:
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
        
        # FAQ мэдээлэл
        if 'faqs' in data and data['faqs']:
            context_parts.append("\n=== ТҮГЭЭМЭЛ АСУУЛТУУД ===")
            for faq in data['faqs']:
                context_parts.append(f"""
Асуулт: {faq.get('q_keywords', '')}
Хариулт: {faq.get('answer', '')[:150]}...
---""")
        
        # Ерөнхий мэдээлэл
        context_parts.append("""
=== БУСАД МЭДЭЭЛЭЛ ===
Хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж
Утас: 91117577, 99201187
Имэйл: hello@wayconsulting.io
Академийн онцлог: Салбарын шилдэг багш нар, Бодит төсөл дээр практик, AI-г сургалтад нэвтрүүлсэн""")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, user_question: str, context_data: Dict[str, Any]) -> str:
        """AI алдаа гарвал fallback хариулт үүсгэх"""
        courses = context_data.get('courses', [])
        faqs = context_data.get('faqs', [])
        
        # Энгийн keyword matching
        user_lower = user_question.lower()
        
        # Түлхүүр үгсээр хайх
        keywords = {
            'сургалт': 'Бидэнд олон төрлийн сургалтууд байна: ',
            'үнэ': 'Сургалтын үнийн мэдээлэл: ',
            'цаг': 'Сургалтын цагийн хуваарь: ',
            'багш': 'Бидний багш нар: ',
            'хаяг': 'Бидний хаяг: Galaxy Tower, 7 давхар, 705 тоот, Махатма Ганди гудамж',
            'утас': 'Утас: 91117577, 99201187',
            'имэйл': 'Имэйл: hello@wayconsulting.io'
        }
        
        for keyword, response in keywords.items():
            if keyword in user_lower:
                if keyword == 'сургалт' and courses:
                    course_names = [c.get('course_name', '') for c in courses[:3]]
                    return f"{response}{', '.join(course_names)}. Дэлгэрэнгүй: 91117577"
                return response
        
        # Хэрэв ямар ч мэдээлэл олдохгүй бол
        return "Уучлаарай, энэ асуултанд хариулж чадахгүй байна. Дэлгэрэнгүй мэдээлэл авах бол 91117577 дугаарт залгана уу."

# ======================
# ManyChat Service
# ======================
class ManyChatService:
    @staticmethod
    def send_message(subscriber_id: str, message: str) -> Dict[str, Any]:
        """ManyChat руу мессеж илгээх"""
        try:
            token = app.config['MANYCHAT_TOKEN']
            if not token:
                logger.warning("⚠️ ManyChat Token олдсонгүй!")
                return {"status": "error", "message": "Token not configured"}
            
            # ManyChat V2 API форматыг засах
            payload = {
                "subscriber_id": subscriber_id,
                "message": message  # Шууд message field-д оруулах
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # API endpoint засах
            response = requests.post(
                app.config['MANYCHAT_API_URL'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                logger.info(f"✅ ManyChat руу амжилттай илгээлээ: {subscriber_id}")
            else:
                logger.warning(f"⚠️ ManyChat алдаатай буцаасан: {result}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ ManyChat руу илгээхэд алдаа: {e}")
            return {"status": "error", "message": str(e)}

# ======================
# Глобал Service Объектууд
# ======================
try:
    sheets_service = GoogleSheetsService()
    ai_service = AIService()
    manychat_service = ManyChatService()
    logger.info("✅ Бүх сервисүүд амжилттай эхэллээ")
except Exception as e:
    logger.error(f"❌ Сервис эхлүүлэхэд алдаа: {e}")
    # Сервисүүдийг None болгох
    sheets_service = None
    ai_service = None
    manychat_service = None

# ======================
# Flask Routes
# ======================
@app.route('/')
def index():
    """Үндсэн хуудас - систем статус харуулах"""
    return jsonify({
        "status": "active",
        "service": "Way Academy Chatbot API",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "endpoints": {
            "/health": "Эрүүл мэндийн шалгалт",
            "/manychat/webhook": "ManyChat вебхук",
            "/test": "Тестийн endpoint",
            "/courses": "Бүх сургалтууд",
            "/faqs": "Бүх FAQ"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Эрүүл мэндийн шалгалт"""
    services_status = {
        "google_sheets": False,
        "openai": False,
        "manychat": bool(app.config['MANYCHAT_TOKEN'])
    }
    
    # Google Sheets шалгалт
    try:
        if sheets_service:
            test_data = sheets_service.get_all_courses()
            services_status["google_sheets"] = len(test_data) > 0
    except Exception as e:
        logger.error(f"❌ Google Sheets шалгалтын алдаа: {e}")
        services_status["google_sheets"] = False
    
    # OpenAI/Groq шалгалт
    services_status["openai"] = bool(app.config['OPENAI_API_KEY'])
    
    overall_status = "healthy" if all(services_status.values()) else "degraded"
    
    return jsonify({
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "services": services_status,
        "version": "2.0.0",
        "message": "Сервис ажиллаж байна" if overall_status == "healthy" else "Зарим сервис ажиллахгүй байна"
    })

@app.route('/manychat/webhook', methods=['POST'])
def manychat_webhook():
    """ManyChat вебхук endpoint"""
    try:
        data = request.json

        if not data:
            return jsonify({"error": "Хоосон хүсэлт"}), 400

        subscriber_id = None
        user_message = ""

        if 'subscriber' in data:
            subscriber_id = data['subscriber'].get('id')
            user_message = data.get('message', {}).get('text', '').strip()
        elif 'subscriber_id' in data:
            subscriber_id = data['subscriber_id']
            user_message = data.get('message', '').strip()
        elif 'data' in data and 'subscriber' in data['data']:
            subscriber_id = data['data']['subscriber'].get('id')
            user_message = data.get('data', {}).get('message', '').strip()
        else:
            logger.warning(f"❌ Unknown ManyChat format: {data.keys()}")
            subscriber_id = "unknown"
            user_message = data.get('text', data.get('message', 'сайн уу')).strip()

        if not subscriber_id:
            return jsonify({"error": "Subscriber ID олдсонгүй"}), 400

        if not user_message:
            user_message = "сайн уу"

        logger.info(f"📩 ManyChat ирсэн мессеж: {user_message[:50]}... (Subscriber: {subscriber_id})")

        all_courses, all_faqs = [], []
        if sheets_service:
            all_courses = sheets_service.get_all_courses()
            all_faqs = sheets_service.get_all_faqs()
        else:
            logger.error("❌ Google Sheets сервис ажиллахгүй байна")

        matched_courses = []
        simple_greetings = ['сайн уу', 'сайн байна уу', 'hello', 'hi', 'сайн', 'байна уу']
        if user_message.lower() not in simple_greetings and sheets_service:
            course = sheets_service.get_course_by_keyword(user_message)
            if course:
                matched_courses = [course]

        context_data = {
            "courses": matched_courses if matched_courses else all_courses[:4],
            "faqs": all_faqs[:5]
        }

        if ai_service:
            ai_response = ai_service.generate_response(user_message, context_data)
        else:
            ai_response = "Уучлаарай, AI сервис ажиллахгүй байна. Та 91117577 дугаарт залгана уу."

        # ✅ FIX: ManyChat sendContent API-р давхар илгээхийг БҮРЭН зогсоов.
        # Учир нь ManyChat External Request нь webhook-ийн response-ийг өөрөө ашиглаж/харуулдаг.
        # manychat_service.send_message(subscriber_id, ai_response)

        # ✅ FIX: ManyChat-ийн "Response mapping" ашиглаж байсан хувилбар + mapping-гүй хувилбар
        # хоёуланд нь нийцүүлэхээр 2 wrapper-тэй буцааж байна.
        return jsonify({
            # mapping ашигладаг бол (өмнөх чинь $.content.messages[0].text)
            "content": {
                "messages": [{
                    "type": "text",
                    "text": ai_response
                }]
            },
            # mapping ашиглахгүй, шууд response-г уншдаг тохиргоонд
            "messages": [{
                "type": "text",
                "text": ai_response
            }]
        })

    except Exception as e:
        logger.error(f"❌ Вебхук боловсруулахад алдаа: {e}", exc_info=True)
        return jsonify({
            "messages": [{
                "type": "text",
                "text": "Уучлаарай, техникийн алдаа гарлаа. Та дахин оролдоно уу эсвэл 91117577 дугаарт залгана уу."
            }]
        }), 500

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Тестийн endpoint"""
    if request.method == 'POST':
        data = request.json
        question = data.get('question', 'дижитал маркетинг сургалт')
        
        # AI тест
        courses = []
        faqs = []
        
        if sheets_service:
            courses = sheets_service.get_all_courses()
            faqs = sheets_service.get_all_faqs()
        
        context = {
            "courses": courses[:2] if courses else [],
            "faqs": faqs[:2] if faqs else []
        }
        
        response = ""
        if ai_service:
            response = ai_service.generate_response(question, context)
        else:
            response = "AI сервис ажиллахгүй байна"
        
        return jsonify({
            "question": question,
            "ai_response": response,
            "courses_count": len(courses) if courses else 0,
            "faqs_count": len(faqs) if faqs else 0,
            "services": {
                "google_sheets": sheets_service is not None,
                "ai_service": ai_service is not None
            }
        })
    
    # GET хүсэлтэд ерөнхий мэдээлэл харуулах
    courses = []
    faqs = []
    
    if sheets_service:
        courses = sheets_service.get_all_courses()
        faqs = sheets_service.get_all_faqs()
    
    return jsonify({
        "total_courses": len(courses) if courses else 0,
        "total_faqs": len(faqs) if faqs else 0,
        "sample_course": courses[0]['course_name'] if courses else "Байхгүй",
        "sample_faq": faqs[0]['q_keywords'] if faqs else "Байхгүй",
        "config": {
            "sheet_id": app.config['SHEET_ID'][:10] + "..." if app.config['SHEET_ID'] else "Байхгүй",
            "has_openai_key": bool(app.config['OPENAI_API_KEY']),
            "has_manychat_token": bool(app.config['MANYCHAT_TOKEN'])
        }
    })

@app.route('/courses', methods=['GET'])
def get_courses():
    """Бүх сургалтуудыг авах API"""
    courses = []
    
    if sheets_service:
        courses = sheets_service.get_all_courses()
    
    # Товчлон харуулах
    simplified = []
    for course in courses:
        simplified.append({
            "id": course.get('course_id', 'N/A'),
            "name": course.get('course_name', 'N/A'),
            "teacher": course.get('teacher', 'N/A'),
            "duration": course.get('duration', 'N/A'),
            "price": course.get('price_full', 'N/A'),
            "discount": course.get('price_discount', 'N/A'),
            "schedule": course.get('schedule_1', 'N/A'),
            "keywords": course.get('keywords', '')
        })
    
    return jsonify({
        "status": "success" if sheets_service else "error",
        "count": len(courses),
        "courses": simplified,
        "message": "" if sheets_service else "Google Sheets сервис ажиллахгүй байна"
    })

@app.route('/faqs', methods=['GET'])
def get_faqs():
    """Бүх FAQ-уудыг авах API"""
    faqs = []
    
    if sheets_service:
        faqs = sheets_service.get_all_faqs()
    
    simplified = []
    for faq in faqs:
        simplified.append({
            "id": faq.get('faq_id', 'N/A'),
            "keywords": faq.get('q_keywords', 'N/A'),
            "answer": faq.get('answer', ''),
            "answer_preview": (faq.get('answer', '')[:100] + '...') if faq.get('answer', '') else ''
        })
    
    return jsonify({
        "status": "success" if sheets_service else "error",
        "count": len(faqs),
        "faqs": simplified,
        "message": "" if sheets_service else "Google Sheets сервис ажиллахгүй байна"
    })

# ======================
# Алдааны боловсруулагч
# ======================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint олдсонгүй", "timestamp": datetime.now().isoformat()}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Server алдаа: {error}")
    return jsonify({"error": "Дотоод серверийн алдаа", "timestamp": datetime.now().isoformat()}), 500

# ======================
# Үндсэн функц
# ======================
if __name__ == '__main__':
    # Шаардлагатай тохиргоог шалгах
    required_envs = {
        'SHEET_ID': 'Google Sheets ID',
        'GOOGLE_CREDENTIALS_JSON': 'Google Service Account JSON'
    }
    
    missing = []
    for env, description in required_envs.items():
        if not os.getenv(env):
            missing.append(f"{env} ({description})")
    
    if missing:
        logger.error(f"❌ Дараах environment variable дутуу байна: {', '.join(missing)}")
        if os.getenv('FLASK_DEBUG', 'False').lower() != 'true':
            logger.error("Production mode дээр дутуу тохиргоотойгоор ажиллах боломжгүй!")
            sys.exit(1)
        else:
            logger.warning("⚠️ Debug mode дээр дутуу тохиргоотойгоор ажиллаж байна...")
    
    # Flask сервер эхлүүлэх
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Way Academy Chatbot Server {port} порт дээр эхэллээ...")
    logger.info(f"📊 Google Sheets ID: {app.config['SHEET_ID'][:10]}...")
    logger.info(f"🤖 AI Model: {app.config['OPENAI_MODEL']}")
    logger.info(f"🔧 Debug Mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=False)