"""
generator.py — AUGMENT + GENERATE: build the grounded prompt and call the LLM.

The LLM runs locally through Ollama (/api/chat), called with urllib from the
standard library — no paid API and no extra Python packages.

generate(question, chunks, history=None) -> answer text (str)
    chunks  : retrieved chunks, numbered [1], [2], ... in the prompt
    history : earlier turns [{"question": "...", "answer": "..."}, ...]
              (only the last 2 are sent to the model)
"""

import json
import urllib.error
import urllib.request

from config import OLLAMA_URL, MODEL_NAME, LLM_TIMEOUT_SECONDS, MAX_TOKENS, NUM_CTX, REFUSAL_TEXT

SYSTEM_PROMPT = f"""أنت "المرشد الأكاديمي الذكي" لطلبة كلية تكنولوجيا المعلومات في جامعة الخليل، لتخصصَي:
بكالوريوس علم الحاسوب، وبكالوريوس علم الحاسوب / فرعي إدارة أعمال.

القواعد:
1. أجب فقط من المقاطع المرقّمة [1] [2] ... المرفقة مع السؤال. لا تستخدم معرفتك العامة ولا تخمّن.
2. اكتب الجواب بجمل كاملة تحتوي المعلومة نفسها (الأرقام والرموز والأسماء)، ثم ضع رقم المقطع بين قوسين مربعين بعد كل معلومة، مثل: [2]. لا تكتب أرقام المقاطع وحدها بلا جواب.
3. اقرأ المقطع بدقة: في وصف المساقات، "رقم المساق" هو رمز المساق نفسه، و"متطلب سابق" هو المساق الذي يجب اجتيازه قبله. لا تخلط بينهما.
4. أجب عن السؤال المطلوب بالضبط. إذا كانت المقاطع تحتوي معلومة قريبة أو عامة لكنها لا تجيب عن السؤال نفسه (مثلاً: موعد امتحان مساق محدد غير مذكور)، فاعتبر الجواب غير موجود.
5. الطالب قد يكتب بالعامية؛ افهم سؤاله وأجب بالعربية الفصحى البسيطة وباختصار.
6. لا تخلط بين التخصصين: إذا كانت المعلومة في المقطع خاصة بتخصص معيّن فاذكر ذلك صراحةً، ولا تنسبها لتخصص آخر.
7. إذا لم يكن الجواب موجوداً في المقاطع، أو كان السؤال خارج الإرشاد الأكاديمي (مثل كتابة شعر أو مواضيع عامة)، فاكتب حرفياً هذه الجملة فقط بلا أي إضافة:
{REFUSAL_TEXT}"""

ANSWER_REMINDER = (
    "تذكّر: أجب من المقاطع فقط بجمل كاملة فيها المعلومة نفسها، وضع رقم المقطع بعد كل معلومة مثل [1]. "
    f"إذا لم تجد الجواب بالضبط في المقاطع فاكتب فقط: {REFUSAL_TEXT}"
)


def format_chunks(chunks):
    """Number the chunks [1], [2], ... with their source and page."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        parts.append(f"[{number}] (المصدر: {chunk['source']}، صفحة {chunk['page']})\n{chunk['text']}")
    return "\n\n".join(parts)


def build_messages(question, chunks, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-2:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({
        "role": "user",
        "content": f"المقاطع:\n\n{format_chunks(chunks)}\n\nسؤال الطالب: {question}\n\n{ANSWER_REMINDER}",
    })
    return messages


def call_ollama(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "keep_alive": "60m",          # keep the model loaded in memory between questions
        "options": {"temperature": 0, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL} - is it running? ({error.reason})"
        ) from error
    return data["message"]["content"].strip()


def warm_up():
    """Load the model into memory with a tiny request, so the first real question is not slow."""
    payload = {"role": "user", "content": "مرحبا"}
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps({"model": MODEL_NAME, "messages": [payload], "stream": False,
                         "keep_alive": "60m", "options": {"num_predict": 1, "num_ctx": NUM_CTX}}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        response.read()


def generate(question, chunks, history=None):
    return call_ollama(build_messages(question, chunks, history))
