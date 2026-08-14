from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_service import analyze_message


app = FastAPI(
    title="Daway AI Assistant API",
    version="1.0"
)


class MessageRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=1000
    )


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

        return {
            "source": "gemini",
            **result
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        print("AI ERROR:", repr(error))

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )