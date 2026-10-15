import logging
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, StringConstraints

from ai_service import analyze_message, AIServiceError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dawaak_api")

app = FastAPI(
    title="Dawaak AI Assistant API",
    version="1.1"
)

API_KEY = os.getenv("DAWAAK_API_KEY") or os.getenv("DAWAY_API_KEY") or os.getenv("API_KEY")


def check_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class MessageRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Dawaak AI Assistant API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/ai/assistant")
def ai_assistant(request: MessageRequest, x_api_key: Annotated[str | None, Header()] = None):
    check_api_key(x_api_key)
    try:
        result = analyze_message(request.message)
        return {"source": "ai", **result}
 
    except RuntimeError:
        # خطأ إعداد (مثل مفتاح غير موجود): نسجله ولا نكشفه للعميل
        logger.exception("Configuration error")
        raise HTTPException(status_code=500, detail="Server configuration error")
 
    except AIServiceError:
        logger.warning("AI service temporarily unavailable", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable, please try again"
        )
 
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error")
 