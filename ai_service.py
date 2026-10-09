
import os
import time
from typing import Optional, Literal

from openai import OpenAI
from pydantic import BaseModel, Field


class AIServiceError(Exception):
    """Raised when the AI provider is temporarily unavailable."""
    pass


# =========================================================
# 1. Pydantic Models
# =========================================================

Intent = Literal[
    "search_medicine",
    "pharmacy_locator",
    "find_alternative",
    "drug_info",
    "symptom_guidance",
    "greeting",
    "unclear",
    "out_of_scope",
]


class ModelExtraction(BaseModel):
    intent: Intent

    drug_name: Optional[str] = None
    symptoms: list[str] = Field(default_factory=list)

    # Message to show the user if clarification is needed
    user_message: str = ""

    # Whether the backend needs the user's location
    requires_location: bool = False

    # Information requested by the user
    requested_information: list[str] = Field(default_factory=list)

    # Extra search filters for the backend
    filters: dict = Field(default_factory=dict)

    # Maximum number of results requested
    result_limit: int = 3

    # Whether the message indicates a possible emergency
    is_emergency: bool = False


class AssistantResult(ModelExtraction):
    # Local guidance, not generated medical advice
    guidance: Optional[str] = None


# =========================================================
# 2. Local Symptom Guidance
# =========================================================

SYMPTOM_GUIDANCE = {
    "ألم في البطن": (
        "حاول الراحة وتجنب الأطعمة التي تزيد الألم. "
        "إذا كان الألم شديدًا أو مفاجئًا، أو ترافق مع قيء مستمر "
        "أو دم، اطلب تقييمًا طبيًا عاجلًا."
    ),
    "صداع": (
        "حاول الراحة في مكان هادئ واشرب كمية مناسبة من السوائل. "
        "إذا كان الصداع مفاجئًا وشديدًا جدًا، أو ترافق مع ضعف "
        "مفاجئ أو اضطراب في الوعي أو الرؤية، اطلب الطوارئ."
    ),
    "حرارة": (
        "راقب درجة الحرارة واحرص على الراحة وشرب السوائل. "
        "اطلب تقييمًا طبيًا إذا استمرت الحرارة أو ازدادت، "
        "أو ظهرت علامات مقلقة."
    ),
    "دوخة": (
        "اجلس أو استلقِ لتجنب السقوط، ولا تقد السيارة أثناء الدوخة. "
        "إذا ترافق العرض مع ألم في الصدر أو إغماء أو ضعف مفاجئ، "
        "اطلب الطوارئ."
    ),
    "تقيؤ": (
        "حاول تناول رشفات صغيرة ومتكررة من السوائل. "
        "اطلب تقييمًا طبيًا إذا استمر التقيؤ، أو ظهر دم، "
        "أو ظهرت علامات الجفاف."
    ),
    "سعال": (
        "احرص على الراحة وتناول السوائل المناسبة. "
        "إذا ترافق السعال مع صعوبة في التنفس أو ألم شديد في الصدر "
        "أو خروج دم، اطلب تقييمًا طبيًا عاجلًا."
    ),
    "ضيق تنفس": (
        "ضيق التنفس قد يكون حالة طارئة. إذا كان شديدًا أو مفاجئًا، "
        "اطلب الإسعاف أو توجّه إلى الطوارئ فورًا."
    ),
}

GENERIC_FALLBACK_GUIDANCE = (
    "لا يمكن تحديد سبب العرض من الرسالة وحدها. "
    "إذا استمر العرض أو ازداد سوءًا، استشر مختصًا صحيًا. "
    "إذا ظهرت علامات خطيرة، اطلب المساعدة الطبية العاجلة."
)


def build_guidance(symptoms: list[str]) -> Optional[str]:
    if not symptoms:
        return None

    parts = [
        SYMPTOM_GUIDANCE.get(
            symptom,
            GENERIC_FALLBACK_GUIDANCE
        )
        for symptom in symptoms
    ]

    # Remove duplicate guidance
    unique_parts = list(dict.fromkeys(parts))

    return " ".join(unique_parts)


# =========================================================
# 3. DeepSeek System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are the intent classification and entity extraction component
of a medicine discovery application.

Your job is ONLY to understand the user's request and return
structured JSON. Do not search databases, invent medicine data,
provide medicine prices, or invent pharmacy locations.

The backend performs database searches, retrieves prices and stock,
and calculates distances.

Return data matching this schema:

{
  "intent": "search_medicine",
  "drug_name": null,
  "symptoms": [],
  "user_message": "",
  "requires_location": false,
  "requested_information": [],
  "filters": {},
  "result_limit": 3,
  "is_emergency": false
}

INTENTS:

1. search_medicine:
   The user wants to find, locate, or check availability of a medicine.

2. pharmacy_locator:
   The user primarily asks for nearby pharmacies without specifying
   a medicine.

3. find_alternative:
   The user explicitly asks for an alternative to a named medicine.
   Do not recommend an alternative yourself.

4. drug_info:
   The user asks about a medicine's uses, general information,
   price, availability details, or other specific medicine information.

5. symptom_guidance:
   The user describes symptoms and seeks general guidance.

6. greeting:
   A simple greeting without a medical or medicine request.

7. unclear:
   The request is too vague to classify reliably.

8. out_of_scope:
   The request is unrelated to this application's supported functions,
   or asks the AI to diagnose a disease or prescribe a treatment.

IMPORTANT CLASSIFICATION RULES:

- "وين بلاقي بنادول؟" -> search_medicine.
- "هل بنادول متوفر؟" -> search_medicine.
- "كم سعر بنادول؟" -> drug_info.
- "شو استخدامات بنادول؟" -> drug_info.
- "بدي أقرب صيدلية" -> pharmacy_locator.
- "وين بلاقي بنادول قريب مني؟" -> search_medicine.
- "بدي أقرب ثلاث صيدليات فيها بنادول" -> search_medicine.
- "بدي بديل لبنادول" -> find_alternative.
- "عندي صداع وحرارة" -> symptom_guidance.
- "مرحبا" -> greeting.
- "بدي دوا" -> unclear.
- Requests for diagnosis or personalized treatment/prescribing
  should not be answered as if a diagnosis were certain.

REQUESTED INFORMATION:

Use one or more of these values when applicable:
- "medicine_details"
- "price"
- "availability"
- "pharmacies"
- "distance"
- "alternatives"
- "general_guidance"

Examples:
- Asking for a medicine's price:
  requested_information = ["price"]
- Asking where to find a medicine:
  requested_information = ["availability", "pharmacies", "distance"]
- Asking about medicine uses:
  requested_information = ["medicine_details"]
- Asking for an alternative:
  requested_information = ["alternatives"]

LOCATION RULES:

- requires_location = true when the request needs nearby results
  or distance calculations.
- If the user asks to find a medicine nearby, set it to true.
- If the user only asks about medicine uses or price, set it to false
  unless their request explicitly requires a location.
- Never invent the user's location.

RESULT LIMIT:

- Use 3 by default for nearby pharmacy or medicine searches.
- If the user explicitly requests a different number, use that number.
- Keep result_limit between 1 and 10.

FILTERS:

- Store only explicit, useful search filters in the filters object.
- Do not invent values.
- Use an empty object when there are no additional filters.

SYMPTOMS:

Normalize colloquial Arabic symptoms to clear Arabic labels.
Examples:
- "راسي بوجعني" -> "صداع"
- "حرارتي مرتفعة" -> "حرارة"
- "بطني بوجعني" -> "ألم في البطن"
- "مش قادر أتنفس" -> "ضيق تنفس"

EMERGENCY:

Set is_emergency = true only when the message describes signs
that may indicate an immediate emergency, such as severe or sudden
difficulty breathing, loss of consciousness, or sudden severe
neurological symptoms.

When is_emergency is true, do not delay emergency guidance
to ask follow-up questions.

OUTPUT RULES:

- Return valid JSON only.
- Do not include Markdown fences or additional commentary.
- Do not fabricate medicine names if none were mentioned.
- Use null for an unknown drug_name.
- Use [] when there are no symptoms or requested information.
- Use {} when there are no filters.
- Use an empty string for user_message unless clarification is needed.
- Do not make up prices, availability, pharmacy names, or distances.
"""


# =========================================================
# 4. DeepSeek Client
# =========================================================

def get_deepseek_client() -> OpenAI:
    """
    Supports the existing Render variables:
    AI_API_KEY
    AI_BASE_URL

    Also supports dedicated DeepSeek variables:
    DEEPSEEK_API_KEY
    DEEPSEEK_MODEL
    """

    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("AI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "DeepSeek API key is not configured. "
            "Set DEEPSEEK_API_KEY or AI_API_KEY."
        )

    base_url = (
        os.getenv("AI_BASE_URL")
        or "https://api.deepseek.com"
    )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(os.getenv("AI_TIMEOUT", "30")),
        max_retries=int(os.getenv("AI_MAX_RETRIES", "2")),
    )


# =========================================================
# 5. Call DeepSeek
# =========================================================

def call_deepseek(message: str) -> ModelExtraction:
    client = get_deepseek_client()

    model_name = (
        os.getenv("DEEPSEEK_MODEL")
        or os.getenv("AI_MODEL")
        or "deepseek-chat"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_json = response.choices[0].message.content

    if not raw_json:
        raise RuntimeError("DeepSeek returned an empty response.")

    return ModelExtraction.model_validate_json(raw_json)


# =========================================================
# 6. Main Analysis Function
# =========================================================

def analyze_message(
    message: str,
    max_retries: int = 3,
) -> dict:
    message = str(message).strip()

    if not message:
        return AssistantResult(
            intent="unclear",
            drug_name=None,
            symptoms=[],
            user_message="وضح طلبك من فضلك.",
            requires_location=False,
            requested_information=[],
            filters={},
            result_limit=3,
            is_emergency=False,
            guidance=None,
        ).model_dump()

    last_error = None

    for attempt in range(max_retries):
        try:
            extraction = call_deepseek(message)

            result = AssistantResult(
                **extraction.model_dump(),
                guidance=None,
            )

            # Emergency handling must take priority
            if result.is_emergency:
                result.guidance = (
                    "قد تكون هذه حالة طارئة. "
                    "اطلب الإسعاف أو توجّه إلى أقرب قسم طوارئ فورًا. "
                    "لا تنتظر رد التطبيق."
                )

            elif result.intent == "symptom_guidance":
                result.guidance = build_guidance(result.symptoms)

                if not result.guidance:
                    result.guidance = GENERIC_FALLBACK_GUIDANCE

            return result.model_dump()

        except Exception as error:
            last_error = error
            error_text = str(error)

            # Retry temporary rate-limit/server errors only
            is_temporary_error = (
                "429" in error_text
                or "503" in error_text
                or "502" in error_text
                or "504" in error_text
            )

            if is_temporary_error and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("Unable to analyze the message.")
