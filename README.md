# المرشد الأكاديمي الذكي — AI Academic Advisor

> **فريق VECTOR MIND** · IEEE RAG Hackathon 2026 · جامعة الخليل، كلية تكنولوجيا المعلومات

**المشكلة:** طلبة علم الحاسوب، وعلم الحاسوب / فرعي إدارة أعمال، يجدون أنظمة الجامعة والخطط الدراسية متفرقة في ملفات PDF وجداول يصعب البحث فيها.
**الحل:** مساعد RAG يجيب عن أسئلة الطالب بالعربية (والعامية) من المستندات فقط، يذكر المصدر ورقم الصفحة، ويرفض عندما لا يجد دليلاً.

**Problem / Solution (EN):** Students struggle to find answers spread across PDF regulations and study-plan tables. VECTOR MIND is a fully local RAG assistant that answers Arabic (incl. colloquial) questions only from official/curated documents, cites file + page, and refuses when evidence is missing. No paid APIs.

---

## التشغيل خطوة بخطوة — Setup (Windows)

```cmd
:: 1) Python environment
cd VECTOR_MIND\backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 2) Embedding model (once, ~490 MB, pinned + SHA-256 checked) -> models\multilingual-e5-small
python download_model.py

:: 3) Local LLM (free) — install Ollama, then pull the model
winget install Ollama.Ollama
ollama pull qwen2.5:7b

:: 4) Run the server (from VECTOR_MIND\)
cd ..
backend\.venv\Scripts\python backend\server.py
```

افتح **http://localhost:8000** — Open **http://localhost:8000**.
بدائل: `python backend\main.py` (واجهة طرفية) · `python backend\test_ask.py` (اختبار الأسئلة).
على جهاز ضعيف: غيّر `MODEL_NAME` في `backend/config.py` إلى `"qwen2.5:3b"` (أسرع، لكنه أقل دقة).

---

## المعمارية — Architecture

**رحلة السؤال:**
1. الطالب يكتب السؤال في `frontend/index.html` → `POST /api/ask` مع `{"question","history"}`.
2. `server.py` يستدعي `ask(question, history)` في `rag_pipeline.py`.
3. **Retrieve:** السؤال (+ السؤال السابق في المتابعة) يُحوَّل لمتجه بـ E5 محلياً، وChroma يعيد أقرب 5 مقاطع.
4. **Generate:** `generator.py` يرسل المقاطع مرقّمة [1]..[5] + آخر محادثتين إلى Qwen 2.5 عبر Ollama (`temperature 0`).
5. **Guard:** رفض إذا لم توجد مقاطع، أو كتب النموذج جملة الرفض، أو لم يذكر أي مقطع [n]، أو كان الجواب أرقاماً فقط.
6. يعود `{"status","answer","sources":[{"source","page"}]}` → الواجهة تعرض الجواب وكروت المصادر، والرفض بلون برتقالي.

| الملف | الوظيفة |
|---|---|
| `backend/config.py` | كل الإعدادات (المسارات، التقطيع، TOP_K، النموذج، جملة الرفض) |
| `backend/document_loader.py` | قراءة PDF بـ pypdf + NFKC، مع الصفحة لكل نص |
| `backend/text_chunker.py` | تقطيع 500 حرف بتداخل 100، داخل الصفحة الواحدة |
| `backend/embedder.py` | multilingual-e5-small عبر ONNX Runtime (محلي، بدون PyTorch) |
| `backend/vector_store.py` | Chroma: التخزين والبحث |
| `backend/download_model.py` | تنزيل نموذج E5 مرة واحدة مع التحقق من SHA-256 |
| `backend/generator.py` | الـ prompt المؤرَّض واستدعاء Ollama `/api/chat` بـ urllib + تسخين النموذج |
| `backend/rag_pipeline.py` | `build_index()` و`ask()` وطبقات الرفض واستخراج المصادر |
| `backend/server.py` | خادم http.server (مكتبة قياسية): يعرض الواجهة و`/api/ask` |
| `backend/main.py` | واجهة طرفية (`exit` / `new`) |
| `backend/test_ask.py` | اختبار شامل لأسئلة محددة |
| `frontend/index.html` | واجهة دردشة عربية RTL، تعمل على الموبايل والوضع الليلي |

---

## القرارات التقنية — Key Decisions

- **E5 بدل النموذج الافتراضي:** `intfloat/multilingual-e5-small` بدل `all-MiniLM-L6-v2`. على نفس الملف العربي و17 سؤالاً عربياً: **Hit@3 = 14/17 مقابل 1/17** (و Hit@1 9/17 مقابل 1/17). النموذج القديم كان يقطع 37 من 48 مقطعاً عند حد التوكنات.
- **تحويل الجداول لجمل:** جدول الخطة الاسترشادية (فصلان جنباً إلى جنب) كان يخلط الفصلين في كل سطر عند الاستخراج. حوّلناه لجمل كاملة: "تخصص علم الحاسوب، السنة الثانية، الفصل الثاني: المساق T2203 برمجة الويب 1، … المتطلب السابق: T3280". وأعدنا توليد ملف الأنظمة بحيث يبقى كل رقم بجانب كلمته: "15 (خمس عشرة) ساعة". التوليد عبر Microsoft Word لأن Edge/Chrome أنتج حروفاً فارسية (ی، ھ) عند الاستخراج.
- **طبقات الرفض:** (1) بحث فارغ → رفض بدون استدعاء النموذج، (2) تعليمات صريحة بجملة رفض حرفية، (3) جواب بلا استشهاد [n] → رفض، (4) جواب فارغ من المحتوى → رفض.
- **المصادر من الـ metadata:** المصادر لا يكتبها النموذج؛ تُؤخذ من metadata المقاطع التي استشهد بأرقامها، فلا يمكن اختراع مصدر.
- **نموذج محلي مجاني:** Qwen 2.5 عبر Ollama، بدون أي API مدفوع وبدون مفاتيح. التسريع: `keep_alive 60m`، تسخين عند بدء الخادم، `NUM_CTX 2048` (أكبر prompt مقاس: 1351 توكن مع 4 مقاطع)، `MAX_TOKENS 250`.

---

## المستندات — Documents (`documents/original/`)

| الملف | المحتوى | المصدر |
|---|---|---|
| `hu_advisory_plan_cs_2022_text.pdf` | الخطة الاسترشادية 2022 — علم الحاسوب (نسخة نصية من الجدول) | ملف الخطة الاسترشادية (الأصل في `processed/`) |
| `hu_advisory_plan_cs_business_2022_text.pdf` | الخطة الاسترشادية 2022 — علم الحاسوب / فرعي إدارة أعمال | https://www.hebron.edu/index.php/it-dep/it-dep1/it-dep1-guideplan.html |
| `hu_study_plan_cs_business_2022_text.pdf` | الخطة الدراسية 2022 — CS/إدارة أعمال: 138 ساعة وتوزيعها | https://www.hebron.edu/index.php/it-dep/it-dep1/it-dep1-plans.html |
| `hu_course_descriptions_cs_all_minors.pdf` | وصف 80 مساقاً لعلم الحاسوب وكل الفروع | غير معروف (يبدو رسمياً) |
| `hu_regulations_compiled_general_text.pdf` | العلامات، المعدل، لوحة الشرف، العبء الدراسي، القبول، التواصل | تجميعي من صفحات hebron.edu (الروابط داخل الملف) |

الملفات المستبعدة وسبب كل واحد (تقويم غير رسمي، ملخصات مكررة، ملف ممسوح ضوئياً…) في **`documents/SOURCES.md`**.

---

## نتائج الاختبار الحقيقية — Real Test Results

**`qwen2.5:7b`، CPU فقط (Intel i5-1235U، 7.7 GB RAM):**

| السؤال | المتوقع | النتيجة |
|---|---|---|
| كم ساعة بقدر أسجل بالفصل؟ | جواب | ✅ "العبء الدراسي العادي للطالب المتفرغ هو 15 ساعة معتمدة في الفصل" — دليل الأنظمة ص3 |
| متى امتحان مادة الخوارزميات؟ | رفض | ✅ رفض |
| اكتب لي قصيدة عن الجامعة | رفض | ✅ رفض |
| كم ساعة تخصص علم الحاسوب؟ (غير موجود في المستندات) | رفض | ✅ رفض |
| كم ساعة علم الحاسوب فرعي إدارة أعمال؟ | جواب | ✅ "مئة وثمانية وثلاثون" — دليل الأنظمة ص4 (بعد إضافة الخطة الدراسية؛ قبلها ❌ رفض) |
| شو المتطلب السابق لمساق برمجة الويب 1؟ | جواب (T3280) | ❌ رفض زائد مع 7b (المقطع الصحيح مسترجع في المرتبة 1). `qwen2.5:3b` أجاب T3280 صحيحاً |

**وقت الجواب بعد التسريع:** 41 ثانية (سؤال الـ 15 ساعة) و37 ثانية (رفض الامتحان). قبل التسريع: 60–174 ثانية.
تجربة `TOP_K = 4` كسرت سؤال الـ 15 ساعة فأُعيد `TOP_K = 5`.

---

## حدود النظام — Limitations

- **بطء النموذج المحلي:** ~40 ثانية للسؤال على CPU بدون GPU.
- **رفض زائد أحياناً:** يرفض 7b بعض الأسئلة رغم استرجاع المقطع الصحيح. 3b أسرع لكنه يخلط بين التخصصين أحياناً.
- **عدد ساعات تخصص علم الحاسوب المنفرد غير موجود في المستندات المتاحة** — النظام يرفض بدل التخمين.
- **الملفات الممسوحة ضوئياً (صور) غير مدعومة** — لا يوجد OCR (دليل المباني مستبعد لهذا السبب).
- لائحة البكالوريوس الرسمية (الغياب، الانسحاب، الإنذار) والتقويم الأكاديمي الرسمي غير متوفرة → هذه الأسئلة تُرفض.
- عتبة المسافة `MAX_DISTANCE` غير مضبوطة بعد؛ الرفض يعتمد على النموذج وطبقات الحماية.

---

## AI Usage Disclosure

استخدمنا **Claude** (Anthropic) كمساعد في كتابة ومراجعة الكود، وفحص المستندات وتجهيزها (التحويل النصي، التحقق من الاستخراج). القرارات التقنية واختبارها من الفريق.
نموذج التوليد في النظام نفسه هو **Qwen 2.5** يعمل محلياً عبر **Ollama** — بدون أي API مدفوع وبدون مفاتيح.

We used Claude as a coding/review assistant and for document preparation; design decisions and testing were done by the team. The system's generator is Qwen 2.5 running locally via Ollama — no paid API, no keys.
