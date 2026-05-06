# Quiz Chat — Design

**Status:** Draft (brainstorm complete, awaiting user spec review)
**Date:** 2026-05-06
**Owner:** danth-ai-swe
**Related:** existing `/chat` endpoint; quiz-intent gate added on 2026-05-06

---

## 1. Problem & user story

> *As a learner, while taking a quiz I want to ask questions or discuss
> concepts so I can clarify doubts in real time.*

A new chat-style endpoint specialised for the quiz-session UX. The endpoint
co-exists with `/chat` and `/chat/stream`; **no existing code is modified**
except a 2-line additive router registration in `loma_rag/api/app.py`.

The endpoint must handle four scenarios driven from a single text prompt:

1. **Discussion** — learner asks a clarifying / conceptual question. Treat
   like a normal LOMA chat *but never reveal which option is correct*.
2. **Answer attempt** — learner types A/B/C/D, 1/2/3/4, or free-text close
   to one of the four option contents. The system maps it to a letter.
3. **End session** — learner types "finish", "kết thúc", "nộp bài", etc.
4. **Hint** — learner types "hint", "gợi ý", "help", etc.

### Out of scope (per spec)

- Skipping a question.
- Opening a new session, changing difficulty, requesting a different
  question set, switching session topic.
- Tracking session state server-side. Frontend owns `hint_shown`,
  `has_answered_any`, `correct_idx`, etc.

### Critical constraint

The AI's discussion response **must not directly reveal, hint at, or narrow
down the correct option**. That kind of help belongs to the Hint button.

---

## 2. Architecture

```
loma_rag/
├── prompt/
│   └── quiz_discussion.py          [NEW] QUIZ_DISCUSSION_SYSTEM prompt
│                                          (LOMA-style + no-leak block);
│                                          QUIZ_ANSWER_PARSER_SYSTEM
│                                          (LLM-fallback for free-text
│                                          answer mapping); optional
│                                          QUIZ_WEB_SYSTEM mirroring
│                                          WEB_SYSTEM with no-leak block.
├── rag/
│   ├── quiz_intent.py              [NEW] pure intent detection:
│                                          - normalize_text()
│                                          - detect_end_session()
│                                          - detect_hint_request()
│                                          - parse_answer_letter()
│                                          - parse_answer_fuzzy()
│                                          - parse_answer_llm()
│   └── quiz_chat.py                [NEW] orchestrators:
│                                          - run_quiz_chat()
│                                          - stream_quiz_chat()
├── model/
│   └── api_models.py               [ADD] QuizOption, QuizChatRequest,
│                                          QuizChatData, QuizChatResponse
├── api/
│   ├── app.py                      [ADD] register quiz_chat_router
│   └── routes/
│       └── quiz_chat.py            [NEW] POST /quiz/chat,
│                                          POST /quiz/chat/stream
tests/
├── test_quiz_chat_intent.py        [NEW] fast unit tests
└── test_quiz_chat_pipeline.py      [NEW] end-to-end tests
```

### Reuse — import only, never modify

`analyze_query`, `is_insurance_topic`, `Retriever.retrieve`, `WebFallback`,
`select_answer_model`, `build_loma_user_prompt`, `build_web_user_prompt`,
`complete_with_sentinel_detect`, `BG_POOL`, `format_loma_citations`,
`format_web_citations`, `LRUCache`, `_off_topic_response` /
`_no_result_response` (these two are private to `pipeline.py` — duplicate
the trivial 2-line bodies here rather than modifying `pipeline.py` to
export them).

**`prep_web_docs` is duplicated** into `quiz_chat.py` (≤6 LOC) for the
same reason: keeps `pipeline.py` untouched.

---

## 3. API surface

### Request — `POST /quiz/chat`

```python
class QuizOption(BaseModel):
    id: str          # "A" | "B" | "C" | "D" — validated
    content: str     # option text

class QuizChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    options: list[QuizOption] = Field(..., min_length=2, max_length=6)
    query: str    = Field(..., min_length=1)
    top_k: int    = Field(7, ge=1, le=20)
    web_k: int    = Field(5, ge=1, le=10)
```

Pydantic validator on `options`:
- All `id`s in `{"A","B","C","D"}`, no duplicates, contiguous from "A".

### Response wrapper (same template as `/chat`)

```python
class QuizChatData(BaseModel):
    path: str                              # "answer" | "hint" | "finish" |
                                            # "loma" | "web" | "no_result" |
                                            # "off_topic" | "unsupported_language"
    intent: str                            # "answer" | "hint" | "finish" |
                                            # "question" | "off_topic" |
                                            # "unsupported_language"
    answer: Optional[str] = None           # "A"|"B"|"C"|"D" only when
                                            # intent == "answer"
    message: str = ""                      # text reply (discussion answer,
                                            # canned off_topic / no_result, etc.)
    citations: list[Citation] = []
    related_nodes: list[GraphNode] = []
    en_search_query: str = ""
    web_search_used: bool = False

class QuizChatResponse(BaseModel):
    success: bool
    data: Optional[QuizChatData] = None
    error: str = ""
```

### Response examples

**intent=answer**
```json
{"success": true,
 "data": {"path": "answer", "intent": "answer", "answer": "B",
          "message": "", "citations": [], "related_nodes": [],
          "en_search_query": "", "web_search_used": false},
 "error": ""}
```

**intent=question (discussion path)**
```json
{"success": true,
 "data": {"path": "loma", "intent": "question", "answer": null,
          "message": "Underwriting là quá trình …",
          "citations": [{"label": "...", "source": "...", "lesson_id": "..."}],
          "related_nodes": [...],
          "en_search_query": "...",
          "web_search_used": false},
 "error": ""}
```

**intent=hint / finish / off_topic / unsupported_language** — same shape;
`answer=null`, `message` empty for hint/finish, canned text for off_topic
and unsupported_language (reuse existing constants).

---

## 4. Intent detection — order & rules

The orchestrator runs the steps below; the first match returns.

### Step 1 — Language gate (on `query`)
- `analyze_query(client, query)` → `(lang_code, en_search_text)`. LRU cached.
- If `lang_code ∉ SUPPORTED_LANGS` → `intent="unsupported_language"`,
  `message=UNSUPPORTED_LANGUAGE_MSG`, return.

### Step 2 — End-session keyword (Scenario 3)

Keyword set:
```
finish, end session, end quiz, submit and finish, done, quit, exit, stop,
kết thúc, kết thúc phiên, kết thúc buổi học, thoát, dừng, nộp bài
```

Match procedure:
1. Normalize: lowercase, collapse whitespace, strip leading/trailing
   punctuation (`.,!?;:"'()[]`).
2. Exact equality with any keyword → match.
3. `difflib.SequenceMatcher(None, normalized, keyword).ratio() ≥ 0.85`
   for the **shortest** keyword variants (e.g., "finish", "done", "quit",
   "exit", "stop", "thoát", "dừng") to handle typos like "kent thúc",
   "ngp bai" — fall through this rule by allowing typo-distance only
   on the multi-word Vietnamese variants.
4. Substring match: "I want to finish" — if any keyword is a token in
   the normalized text (whole-word) → match.

→ `intent="finish"`, `path="finish"`. Return.

### Step 3 — Hint keyword (Scenario 4)

Keyword set:
```
hint, hin, give me a hint, show me a hint, help, can i have a hint,
gợi ý, gợi y, goi y, cho tôi gợi ý, cho toi mot hint
```
Same match procedure as Step 2. → `intent="hint"`, `path="hint"`. Return.

### Step 4 — Answer parsing (Scenario 2) — hybrid

**4a. Letter/digit exact (always runs, lang-agnostic).**
Regex (case-insensitive, after normalize):
- `^[abcd]$` → that letter uppercased
- `^[1-4]$` → map `{1→A, 2→B, 3→C, 4→D}`
- `^(option\s+|đáp\s*án\s+|câu\s+|answer\s+)?[(\[]?([abcd1-4])[)\].\s]*$`
  catches "Option A", "đáp án B", "(C)", "D.", "1.", etc.

→ `intent="answer"`, `answer=<letter>`. Return.

**4b. Fuzzy match against option contents (lang-gated).**
Only runs if `lang_code(query) == question_lang`. To get
`question_lang`: `analyze_query(client, question)[0]` (LRU cached, hit
rate high for the same quiz question).

For each `option.content`:
- Normalize both sides (lower, strip punctuation, collapse whitespace).
- `score = difflib.SequenceMatcher(None, query_norm, option_norm).ratio()`

If `best_score ≥ 0.75` AND `best - second_best ≥ 0.15`:
→ `intent="answer"`, `answer=<best.id>`. Return.

**4c. LLM fallback (lang-gated, same condition as 4b).**
One call to `azure.detect_model` with `QUIZ_ANSWER_PARSER_SYSTEM`:
> System: "You map the user's reply to one of four multiple-choice
> options or to NO. The user is taking a quiz; the question and four
> options are listed. Reply with exactly A, B, C, D, or NO. Reply NO
> if the user is asking a question, requesting help, or not picking
> an option. No other text."
> User: `Question: <question>\nA: <A>\nB: <B>\nC: <C>\nD: <D>\nUser: <query>`

Cache LRU keyed on `(question_hash, query_hash)` — repeated same prompts
during a session are free.

If response in `{A,B,C,D}` → `intent="answer"`. If `NO` or parse fail
→ fall through (fail-closed).

### Step 5 — Off-topic gate
`is_insurance_topic(client, query)` returns False → `intent="off_topic"`,
`path="off_topic"`, `message=_off_topic_response(lang_code)`. Return.

### Step 6 — Discussion (Scenario 1)

The discussion pipeline (mirrors `run_query` but uses
`QUIZ_DISCUSSION_SYSTEM`):

1. `Retriever.retrieve(query, top_k, user_lang=lang_code, search_text=en_search_text)`.
2. Build LOMA user prompt via `build_loma_user_prompt(retr, user_lang)`.
3. Speculative web prep via `BG_POOL.submit(prep_web_docs, ...)` if
   confidence is mid-range — same gating thresholds as `run_query`
   (`EARLY_EXIT_*`, `HIGH_CONFIDENCE_*`).
4. `complete_with_sentinel_detect(chat_client, QUIZ_DISCUSSION_SYSTEM,
   user_prompt, model=select_answer_model(query))`.
5. Sentinel hit → fall back to web docs (use `QUIZ_WEB_SYSTEM`, mirrors
   `WEB_SYSTEM` rules + no-leak block).
6. Empty web docs → `path="no_result"`, `message=_no_result_response(lang)`.
7. Otherwise → `intent="question"`, `path="loma"|"web"`, `message=<text>`,
   citations, related_nodes, en_search_query.

---

## 5. No-leak guardrail (Scenario 1)

`QUIZ_DISCUSSION_SYSTEM` extends the LOMA system prompt rules with this
constraint block prepended:

```
You are helping a learner during a live quiz session. The current quiz
question and its multiple-choice options are shown below.

QUIZ QUESTION:
<question>

OPTIONS:
A. <option A>
B. <option B>
C. <option C>
D. <option D>

CRITICAL CONSTRAINTS — these override every other instruction:
1. Do NOT state, hint, imply, or even narrow down which option (A/B/C/D)
   is correct. Do NOT rank the options. Do NOT eliminate options.
2. Do NOT solve the quiz question for the learner.
3. You MAY explain the underlying concept, definitions, or related ideas
   in general terms — without referencing the specific options.
4. If the learner directly asks "which one is correct?" or similar,
   politely decline and remind them to use the Hint button instead.
5. Answer in the learner's language (same rule as normal LOMA chat).
6. Cite chunks with [chunk_id] markers exactly as in the LOMA system rules.
```

The question/options block is interpolated at request time; the
constraints are static. `QUIZ_WEB_SYSTEM` is the same constraint block
prefix applied on top of the existing `WEB_SYSTEM` rules.

Smoke test for leak: assert the rendered discussion `message` does NOT
contain phrases like "the answer is", "correct option is", "đáp án đúng",
"đáp án là", "answer:", a single bare letter on its own line, etc. This
is a heuristic guard — primary defence is the system prompt.

---

## 6. Streaming endpoint (`POST /quiz/chat/stream`)

Reuses the SSE shape of `/chat/stream`:

| Event | When |
|---|---|
| `{"type":"stage","stage":"analyze_query"}` | always first |
| `{"type":"stage","stage":"intent_detected","intent":"finish"\|"hint"\|"answer"\|"off_topic"\|"unsupported_language"\|"question"}` | once decided |
| `{"type":"delta","text":"..."}` | only for `intent="question"` while LLM streams |
| `{"type":"done", ... }` | terminal — full data dict identical to non-streaming response |

For non-discussion intents, the route emits `analyze_query` →
`intent_detected` → `done` with no `delta` events.

---

## 7. Edge cases & error handling

| Case | Behaviour |
|---|---|
| `options` empty / fewer than 2 / > 6 | 422 (pydantic) |
| Duplicate or non-A/B/C/D ids | 422 (validator) |
| `query`, `question` empty / whitespace | 422 |
| `query == "A B"` (ambiguous) | letter-exact regex fails (multi-token) → fuzzy/LLM may resolve, otherwise discussion |
| Lang mismatch on free-text answer (e.g., FR query, EN question) | Step 1 still passes if FR is supported (it isn't — fr ∉ {vi,en,ja} → unsupported_language). For lang in supported set but ≠ question_lang, skip 4b/4c, then off-topic gate, then discussion |
| LLM answer-parser exception | Treat as `NO` (fail-closed) — fall through |
| Retriever exception | `success=false`, `data=null`, `error="<Type>: <msg>"` (mirror `/chat` pattern, but return wrapped — never raise HTTP 500) |
| Web fallback exception | `intent="question"`, `path="no_result"`, `message=canned`, success=true |
| Sentinel hit but `web_fallback is None` | `path="refused"`, `message=""`, intent stays `"question"` |

Never raise HTTPException. Always return the wrapped envelope so the
frontend's response handler is uniform.

---

## 8. Testing

### `tests/test_quiz_chat_intent.py` (fast — no retriever / web)

Direct calls to `quiz_intent` pure functions. Approx case counts:

- **End-session detection (~12 cases):**
  positive — "Kết thúc", "end session", "nộp bài", "kent thúc",
  "I want to finish", "DONE", "submit and finish";
  negative — "How do I end a contract?", "what does 'finish' mean?",
  "stop the misclassification" (non-quiz context).
- **Hint detection (~10 cases):**
  positive — "hint", "Hint!", "gợi ý", "Help", "cho tôi 1 hint",
  "goi y", "show me a hint";
  negative — "what does hint mean in poker?", "I'd rather not have a
  hint, just an example" (token-containment will misfire; accepted
  v1 false-positive — see Open items).
- **Answer parsing — letter/digit exact (~10):**
  "A", "a", "B.", "(C)", "1", "Option D", "đáp án B", "Câu A",
  "answer: c", "  D  ".
- **Answer parsing — fuzzy (~6):**
  query close to option content, same lang as question — positive;
  negative cases where best-second_best gap is too small.
- **Answer parsing — lang-mismatch (~3):**
  VI query, EN question → expect None (fall through).
- **Answer parsing — LLM-needed paraphrase (~4):**
  Marked `requires_llm=True`, skipped if env flag is unset.
- **Conflict cases:**
  "A is the answer right?" → expect intent NOT answer (it's a question).

Pattern follows `tests/test_quiz_intent.py`: `CASES` list, runner prints
pass/fail count, exit 0/1.

### `tests/test_quiz_chat_pipeline.py` (slow — full pipeline)

End-to-end through `run_quiz_chat()`:

- **2 discussion cases (intent=question):** smoke test that `message`
  does not contain "the answer is", "correct option", "đáp án đúng",
  "đáp án là", "answer:" (heuristic).
- **1 answer case (intent=answer):** `data.answer == expected letter`.
- **1 finish case (intent=finish):** `data.intent == "finish"`,
  `data.answer is None`, `data.message == ""`.
- **1 hint case (intent=hint):** same shape.
- **1 off-topic during quiz:** intent=off_topic, message=canned.

Runner pattern: same as `tests/test_qa.py`.

---

## 9. Open items / non-goals

- Server-side session storage — explicitly out of scope. Frontend owns
  state.
- Hint content generation — out of scope. The endpoint only signals
  intent. The frontend decides what hint to display and tracks the
  one-time-only rule.
- Correctness feedback — out of scope. The frontend has the correct
  answer and grades.
- Streaming for non-discussion intents — they finish in <1 LLM call;
  not worth chunking.
- Tuning the fuzzy threshold (0.75 / 0.15) — start with these values,
  revisit if test cases reveal misclassification.
- Token-containment false positives for end-session / hint keywords
  (e.g., "Don't finish this", "I'd rather not have a hint"). Accepted
  v1 risk — these phrasings are rare in a quiz UX and the LLM
  no-leak guardrail still applies if they slip into discussion.

---

## 10. Acceptance criteria

1. `POST /quiz/chat` accepts `QuizChatRequest` and returns
   `QuizChatResponse` matching the shapes above.
2. `POST /quiz/chat/stream` emits SSE in the documented shape.
3. All four intents are detected correctly on the unit test suite.
4. The discussion path's `message` passes the no-leak heuristic on the
   pipeline test suite.
5. No file under `loma_rag/` is modified except:
   - `loma_rag/api/app.py` — additive `include_router` line.
   - `loma_rag/model/api_models.py` — additive new classes.
6. `/chat` and `/chat/stream` continue to behave identically (the QA
   suite from `tests/test_qa.py` still passes).
