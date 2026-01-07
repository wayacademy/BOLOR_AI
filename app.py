import os
import json
import logging
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
logging.basicConfig(level=logging.INFO)
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
    
    # OpenAI API
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # ManyChat тохиргоо
    MANYCHAT_TOKEN = os.getenv('MANYCHAT_TOKEN')
    MANYCHAT_API_URL = "https://api.manychat.com/fb/sending/sendContent"
    
    # Кэш хугацаа (секундээр)
    CACHE_TTL = 300  # 5 минут

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
            credentials_info = json.loads(app.config['CREDENTIALS_JSON'])
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            self.service = build('sheets', 'v4', credentials=credentials)
            logger.info("✅ Google Sheets API сервис эхлэв")
        except Exception as e:
            logger.error(f"❌ Google Sheets API эхлүүлэхэд алдаа: {e}")
            raise
    
    @lru_cache(maxsize=1)
    def get_cached_data(self, sheet_name: str, cache_key: str = ""):
        """Кэшлэсэн өгөгдөл авах"""
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
        keyword_lower = keyword.lower()
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
        
        return None

# ======================
# AI Service (OpenAI)
# ======================
class AIService:
    def __init__(self):
        openai.api_key = app.config['OPENAI_API_KEY']
        self.model = app.config['OPENAI_MODEL']
        
        if not openai.api_key:
            logger.warning("⚠️ OpenAI API Key олдсонгүй!")
    
    def generate_response(self, user_question: str, context_data: Dict[str, Any]) -> str:
        """AI ашиглан хариулт үүсгэх"""
        try:
            if not openai.api_key:
                return "Уучлаарай, AI сервис түр ажиллахгүй байна."
            
            # Монгол хэл дээрх систем prompt
            system_prompt = """Та бол Way Academy-гийн албан ёсны туслах чатбот. 
Дараах дүрмийг баримтлаарай:
1. ЗӨВХӨН өгөгдсөн мэдээллээс хариулт өгөх
2. Монгол хэлээр, найрсаг, товч хариулт өгөх
3. Үнэ, цаг, багшийн мэдээллийг тодорхой харуулах
4. Хэрэв мэдээлэл олдохгүй бол "Уучлаарай, энэ асуултанд хариулж чадахгүй байна" гэж хэлэх
5. Сургалтын нэр, ID (SDM, DA гэх мэт) зөв хэрэглэх"""
            
            # Context-ыг форматлах
            context_str = self._format_context(context_data)
            
            # Хэрэглэгчийн асуулт
            user_prompt = f"""Хэрэглэгчийн асуулт: {user_question}

Доорх мэдээллээс хариулт өгнө үү:
{context_str}

Хариулт:"""
            
            # OpenAI API дуудах
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ AI хариулт үүсгэхэд алдаа: {e}")
            return "Уучлаарай, хариулт үүсгэхэд алдаа гарлаа. Дахин оролдоно уу."
    
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
            
            payload = {
                "subscriber_id": subscriber_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "messages": [{
                            "type": "text",
                            "text": message
                        }]
                    }
                }
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(
                app.config['MANYCHAT_API_URL'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            logger.info(f"✅ ManyChat руу амжилттай илгээлээ")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ ManyChat руу илгээхэд алдаа: {e}")
            return {"status": "error", "message": str(e)}

# ======================
# Глобал Service Объектууд
# ======================
sheets_service = GoogleSheetsService()
ai_service = AIService()
manychat_service = ManyChatService()

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
        test_data = sheets_service.get_all_courses()
        services_status["google_sheets"] = len(test_data) > 0
    except:
        services_status["google_sheets"] = False
    
    # OpenAI шалгалт
    services_status["openai"] = bool(app.config['OPENAI_API_KEY'])
    
    return jsonify({
        "status": "healthy" if all(services_status.values()) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": services_status,
        "version": "1.0.0"
    })

@app.route('/manychat/webhook', methods=['POST'])
def manychat_webhook():
    """ManyChat вебхук endpoint"""
    try:
        data = request.json
        
        # Шаардлагатай талбарууд шалгах
        if not data or 'subscriber_id' not in data or 'message' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        subscriber_id = data['subscriber_id']
        user_message = data['message'].strip()
        
        logger.info(f"📩 ManyChat ирсэн мессеж: {user_message[:50]}...")
        
        # 1. Google Sheets-ээс өгөгдөл татах
        all_courses = sheets_service.get_all_courses()
        all_faqs = sheets_service.get_all_faqs()
        
        # 2. Хэрэглэгчийн асуултад тохирох сургалтыг олох
        matched_courses = []
        if user_message:
            course = sheets_service.get_course_by_keyword(user_message)
            if course:
                matched_courses = [course]
        
        # 3. AI хариулт үүсгэх
        context_data = {
            "courses": matched_courses if matched_courses else all_courses[:4],
            "faqs": all_faqs[:5]
        }
        
        ai_response = ai_service.generate_response(user_message, context_data)
        
        # 4. ManyChat руу илгээх
        manychat_response = manychat_service.send_message(subscriber_id, ai_response)
        
        # 5. ManyChat-д шаардлагатай формат буцаах
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
        logger.error(f"❌ Вебхук боловсруулахад алдаа: {e}")
        return jsonify({
            "version": "v2",
            "content": {
                "messages": [{
                    "type": "text",
                    "text": "Уучлаарай, техникийн алдаа гарлаа. Та дахин оролдоно уу эсвэл 91117577 дугаарт залгана уу."
                }]
            }
        }), 500

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Тестийн endpoint"""
    if request.method == 'POST':
        data = request.json
        question = data.get('question', 'дижитал маркетинг сургалт')
        
        # AI тест
        courses = sheets_service.get_all_courses()
        faqs = sheets_service.get_all_faqs()
        
        context = {
            "courses": courses[:2],
            "faqs": faqs[:2]
        }
        
        response = ai_service.generate_response(question, context)
        
        return jsonify({
            "question": question,
            "ai_response": response,
            "courses_count": len(courses),
            "faqs_count": len(faqs)
        })
    
    # GET хүсэлтэд ерөнхий мэдээлэл харуулах
    courses = sheets_service.get_all_courses()
    faqs = sheets_service.get_all_faqs()
    
    return jsonify({
        "total_courses": len(courses),
        "total_faqs": len(faqs),
        "sample_course": courses[0]['course_name'] if courses else None,
        "sample_faq": faqs[0]['q_keywords'] if faqs else None
    })

@app.route('/courses', methods=['GET'])
def get_courses():
    """Бүх сургалтуудыг авах API"""
    courses = sheets_service.get_all_courses()
    
    # Товчлон харуулах
    simplified = []
    for course in courses:
        simplified.append({
            "id": course.get('course_id'),
            "name": course.get('course_name'),
            "teacher": course.get('teacher'),
            "duration": course.get('duration'),
            "price": course.get('price_full'),
            "discount": course.get('price_discount'),
            "schedule": course.get('schedule_1')
        })
    
    return jsonify({
        "count": len(courses),
        "courses": simplified
    })

@app.route('/faqs', methods=['GET'])
def get_faqs():
    """Бүх FAQ-уудыг авах API"""
    faqs = sheets_service.get_all_faqs()
    
    simplified = []
    for faq in faqs:
        simplified.append({
            "id": faq.get('faq_id'),
            "keywords": faq.get('q_keywords'),
            "answer_preview": faq.get('answer', '')[:100] + '...'
        })
    
    return jsonify({
        "count": len(faqs),
        "faqs": simplified
    })

# ======================
# Алдааны боловсруулагч
# ======================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint олдсонгүй"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Server алдаа: {error}")
    return jsonify({"error": "Дотоод серверийн алдаа"}), 500

# ======================
# Үндсэн функц
# ======================
if __name__ == '__main__':
    # Шаардлагатай тохиргоог шалгах
    required_envs = ['SHEET_ID', 'OPENAI_API_KEY', 'GOOGLE_CREDENTIALS_JSON']
    missing = [env for env in required_envs if not os.getenv(env)]
    
    if missing:
        logger.warning(f"⚠️ Дараах environment variable дутуу байна: {missing}")
        logger.warning("Үйлчилгээ дутуу тохиргоотойгоор эхлэв...")
    
    # Flask сервер эхлүүлэх
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Way Academy Chatbot Server {port} порт дээр эхэллээ...")
    logger.info(f"📊 Google Sheets ID: {app.config['SHEET_ID']}")
    logger.info(f"🤖 OpenAI Model: {app.config['OPENAI_MODEL']}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 