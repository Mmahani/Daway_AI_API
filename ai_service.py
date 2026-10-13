
import os
import time
import logging
from typing import Optional, Literal

from openai import OpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger("daway_ai_service")


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

    # Clarification message, when needed
    user_message: str = ""

    # Backend location requirement
    requires_location: bool = False

    # Requested information
    requested_information: list[str] = Field(default_factory=list)

    # Explicit search filters
    filters: dict = Field(default_factory=dict)

    # Maximum requested results
    result_limit: int = Field(default=3, ge=1, le=10)

    # Possible emergency
    is_emergency: bool = False


class AssistantResult(ModelExtraction):
    # Exact original message, set locally rather than by the LLM
    raw_query: str

    # Text suitable for display in the chat
    reply: Optional[str] = None

    # Local general guidance
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
        "اطلب الإسعاف أو توجّه إلى أقرب قسم طوارئ فورًا."
    ),
}

# Common variants mapped to the canonical local guidance topic
SYMPTOM_ALIASES = {
    "كحة": "سعال",
    "كحّة": "سعال",
    "كحه": "سعال",
    "سعال": "سعال",
    "وجع راس": "صداع",
    "وجع رأس": "صداع",
    "راسي بوجعني": "صداع",
    "صداع": "صداع",
    "سخونة": "حرارة",
    "حراره": "حرارة",
    "حرارتي مرتفعة": "حرارة",
    "حرارة": "حرارة",
    "استفراغ": "تقيؤ",
    "ترجيع": "تقيؤ",
    "تقيؤ": "تقيؤ",
    "وجع بطن": "ألم في البطن",
    "بطني بتوجعني": "ألم في البطن",
    "ألم في البطن": "ألم في البطن",
    "ضيق النفس": "ضيق تنفس",
    "مش قادر أتنفس": "ضيق تنفس",
    "ضيق تنفس": "ضيق تنفس",
}

GENERIC_FALLBACK_GUIDANCE = (
    "لا يمكن تحديد سبب العرض من الرسالة وحدها. "
    "إذا استمر العرض أو ازداد سوءًا، استشر مختصًا صحيًا. "
    "إذا ظهرت علامات خطيرة، اطلب المساعدة الطبية العاجلة."
)

EMERGENCY_REPLY = (
    "قد تكون الأعراض التي وصفتها طارئة. إذا كنت تعاني ضيقًا شديدًا "
    "أو مفاجئًا في التنفس، أو فقدانًا للوعي، أو أعراضًا خطيرة مفاجئة، "
    "اطلب الإسعاف أو توجّه إلى أقرب قسم طوارئ فورًا. "
    "لا تنتظر رد التطبيق."
)


def build_guidance(symptoms: list[str]) -> Optional[str]:
    if not symptoms:
        return None

    parts = []

    for symptom in symptoms:
        canonical = SYMPTOM_ALIASES.get(symptom, symptom)
        guidance = SYMPTOM_GUIDANCE.get(
            canonical,
            GENERIC_FALLBACK_GUIDANCE,
        )
        parts.append(guidance)

    # Remove duplicate guidance
    unique_parts = list(dict.fromkeys(parts))
    return " ".join(unique_parts)


# =========================================================
# 3. Local Chat Replies
# =========================================================

def build_reply(result: AssistantResult) -> Optional[str]:
    """
    Builds a user-facing reply for intents that can be acknowledged
    locally. Medicine facts, prices, availability, and pharmacy results
    must still come from the backend.
    """

    if result.is_emergency:
        return result.guidance or EMERGENCY_REPLY

    if result.intent == "greeting":
        return (
            "أهلًا وسهلًا! أنا مساعد دوائي. "
            "كيف بقدر أساعدك اليوم؟"
        )

    if result.intent == "unclear":
        return result.user_message or (
            "أكيد، بساعدك. ممكن توضح طلبك أكثر؟ "
            "هل تريد البحث عن دواء، أو معرفة معلومات عنه، "
            "أم تسأل عن أعراض صحية؟"
        )

    if result.intent == "out_of_scope":
        return (
            "أنا مساعد دوائي، وبقدر أساعدك في البحث عن الأدوية "
            "ومعلوماتها العامة والإرشادات الصحية العامة. "
            "لا أستطيع تشخيص الأمراض أو وصف علاج شخصي."
        )

    if result.intent == "symptom_guidance":
        return result.guidance or GENERIC_FALLBACK_GUIDANCE

    if result.intent == "search_medicine":
        if result.drug_name:
            return (
                f"فهمت أنك تريد البحث عن {result.drug_name}. "
                "سأحتاج إلى نتائج البحث من النظام لعرض المعلومات المتاحة."
            )
        return "ما اسم الدواء الذي تريد البحث عنه؟"

    if result.intent == "pharmacy_locator":
        return (
            "سأحتاج إلى موقعك أو المنطقة التي تريد البحث فيها "
            "حتى يتمكن النظام من تحديد الصيدليات القريبة."
        )

    if result.intent == "find_alternative":
        if result.drug_name:
            return (
                f"فهمت أنك تبحث عن بديل لـ {result.drug_name}. "
                "يجب التحقق من البدائل المناسبة ومراجعة مختص صحي "
                "قبل تغيير الدواء."
            )
        return "ما اسم الدواء الذي تريد معرفة بدائله؟"

    if result.intent == "drug_info":
        if result.drug_name:
            return (
                f"فهمت أنك تريد معلومات عن {result.drug_name}. "
                "سنعتمد على بيانات الدواء الموثوقة المتاحة في النظام."
            )
        return "ما اسم الدواء الذي تريد معرفة معلومات عنه؟"

    return "لم أتمكن من تحديد طلبك بدقة. وضّح طلبك من فضلك."


# =========================================================
# 4. DeepSeek System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are the intent classification and entity extraction component
of a medicine discovery application.

Your task is to understand the user's request and return structured
JSON only. Do not search databases, invent medicine data, provide
medicine prices, or invent pharmacy locations.

The backend performs database searches, retrieves prices and stock,
and calculates distances.

Return JSON matching these fields:
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
- search_medicine: find, locate, or check availability of a medicine.
- pharmacy_locator: nearby pharmacies without a specified medicine.
- find_alternative: user explicitly asks for an alternative.
- drug_info: asks for medicine uses or other specific medicine details.
- symptom_guidance: describes symptoms and seeks general guidance.
- greeting: simple greeting without a medical request.
- unclear: request is too vague to classify reliably.
- out_of_scope: unrelated requests, diagnosis, or personal prescribing.

CLASSIFICATION:
- "وين بلاقي بنادول؟" -> search_medicine
- "هل بنادول متوفر؟" -> search_medicine
- "كم سعر بنادول؟" -> drug_info
- "شو استخدامات بنادول؟" -> drug_info
- "بدي أقرب صيدلية" -> pharmacy_locator
- "وين بلاقي بنادول قريب مني؟" -> search_medicine
- "بدي بديل لبنادول" -> find_alternative
- "عندي صداع وحرارة" -> symptom_guidance
- "مرحبا" -> greeting
- "بدي دوا" -> unclear

REQUESTED INFORMATION:
Use applicable values:
- medicine_details
- price
- availability
- pharmacies
- distance
- alternatives
- general_guidance

LOCATION:
Set requires_location=true when nearby results or distance calculations
are requested. Never invent the user's location.
For medicine uses or price alone, normally set it to false.

RESULT LIMIT:
Default 3. Respect an explicit requested number between 1 and 10.

FILTERS:
Only explicit, useful search filters. Do not invent values.

SYMPTOMS:
Normalize colloquial Arabic symptoms to canonical Arabic labels:
- "راسي بوجعني" -> "صداع"
- "حرارتي مرتفعة" -> "حرارة"
- "بطني بوجعني" -> "ألم في البطن"
- "مش قادر أتنفس" -> "ضيق تنفس"
- "كحة" or "كحّة" -> "سعال"

EMERGENCY:
Set is_emergency=true for signs that may indicate an immediate emergency,
including severe or sudden difficulty breathing, loss of consciousness,
or sudden severe neurological symptoms.
Do not delay emergency guidance to ask follow-up questions.

OUTPUT:
- Return valid JSON only, without Markdown.
- Do not fabricate medicine names when none were mentioned.
- Use null for unknown drug_name.
- Use [] when no symptoms or requested information are present.
- Use {} when there are no filters.
- Keep user_message empty unless clarification is needed.
- Do not invent prices, availability, pharmacy names, or distances.
"""


# =========================================================
# 5. DeepSeek Client
# =========================================================

def get_deepseek_client() -> OpenAI:
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
# 6. Call DeepSeek
# =========================================================

def call_deepseek(message: str) -> ModelExtraction:
    client = get_deepseek_client()

    model_name = (
        os.getenv("DEEPSEEK_MODEL")
        or os.getenv("AI_MODEL")
        or "deepseek-chat"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw_json = response.choices[0].message.content

        if not raw_json:
            raise AIServiceError("AI provider returned an empty response.")

        return ModelExtraction.model_validate_json(raw_json)

    except AIServiceError:
        raise
    except Exception as error:
        error_text = str(error)
        logger.exception("DeepSeek request failed")

        if any(code in error_text for code in ("429", "502", "503", "504")):
            raise AIServiceError(
                "AI provider is temporarily unavailable."
            ) from error

        raise


# =========================================================
# 7. Main Analysis Function
# =========================================================

def analyze_message(
    message: str,
    max_retries: int = 3,
) -> dict:
    raw_query = str(message)
    normalized_message = raw_query.strip()

    if not normalized_message:
        result = AssistantResult(
            intent="unclear",
            drug_name=None,
            symptoms=[],
            user_message="وضح طلبك من فضلك.",
            requires_location=False,
            requested_information=[],
            filters={},
            result_limit=3,
            is_emergency=False,
            raw_query=raw_query,
            reply=None,
            guidance=None,
        )
        result.reply = build_reply(result)
        return result.model_dump()

    last_error = None

    for attempt in range(max_retries):
        try:
            extraction = call_deepseek(normalized_message)

            result = AssistantResult(
                **extraction.model_dump(),
                raw_query=raw_query,
                reply=None,
                guidance=None,
            )

            # Emergency handling takes priority
            if result.is_emergency:
                result.guidance = EMERGENCY_REPLY

            elif result.intent == "symptom_guidance":
                result.guidance = build_guidance(result.symptoms)

                if not result.guidance:
                    result.guidance = GENERIC_FALLBACK_GUIDANCE

            # For unclear requests, always provide a clarification prompt
            if result.intent == "unclear" and not result.user_message.strip():
                result.user_message = (
                    "وضح طلبك من فضلك. هل تريد البحث عن دواء، "
                    "أو معرفة معلومات عنه، أم تسأل عن أعراض صحية؟"
                )

            # Create a user-facing reply locally
            result.reply = build_reply(result)

            return result.model_dump()

        except AIServiceError:
            raise

        except Exception as error:
            last_error = error
            error_text = str(error)

            is_temporary_error = any(
                code in error_text
                for code in ("429", "502", "503", "504")
            )

            if is_temporary_error and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("Unable to analyze the message.")
