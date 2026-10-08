import logging
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
 
from ai_service import analyze_message, AIServiceError
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("daway_api")
 
app = FastAPI(
    title="Daway AI Assistant API",
    version="1.1"
)
 
 
class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
 
 
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Daway AI Assistant API is running"
    }
 
 
@app.post("/ai/assistant")
def ai_assistant(request: MessageRequest):
    try:
        result = analyze_message(request.message)
        return {"source": "ai", **result}
 
    except RuntimeError:
        # خطأ إعداد (مثل مفتاح غير موجود): نسجله ولا نكشفه للعميل
        logger.exception("Configuration error")
        raise HTTPException(status_code=500, detail="Server configuration error")
 
    except AIServiceError:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable, please try again"
        )
 
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error")
 