"""Q&A test cases for the LOMA RAG chatbot.

Each case is a dict with:
  id           - unique short label
  category     - bucket for filtering / reporting
  query        - the user question
  expects      - dict of assertions:
      path                       "loma" | "web" | "refused" (any of these)
      language                   "vi" | "en"  (heuristic check on answer)
      must_contain               list[str]  case-insensitive substrings (all must appear)
      must_contain_any           list[str]  any one substring must appear
      must_cite_pattern          regex; must match somewhere in answer_text
      must_cite_lesson_starting  prefix; at least one cited LOMA chunk's lesson_id must start with this
      min_loma_chunks            int; at least this many chunks retrieved (path=="loma" only)
      min_web_docs               int; at least this many web docs (path=="web" only)
      min_related_nodes          int; minimum graph-expanded nodes

Add or edit cases freely; the runner pulls them from the TESTS list.
"""
from __future__ import annotations

TESTS: list[dict] = [
    # ===== concept definition - Vietnamese =====
    {
        "id": "vi_def_antiselection",
        "category": "concept_vi",
        "query": "Antiselection là gì? Tại sao underwriter cần quan tâm tới nó?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["antiselection", "lựa chọn"],
            "must_cite_lesson_starting": "LOMA281_M1L1",
        },
    },
    {
        "id": "vi_def_longevity",
        "category": "concept_vi",
        "query": "Longevity risk có nghĩa là gì trong bảo hiểm?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["longevity", "outliv", "tuổi thọ", "tài chính"],
        },
    },
    {
        "id": "vi_def_reinsurance",
        "category": "concept_vi",
        "query": "Reinsurance là gì và ai nhận rủi ro?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["reinsurer", "reinsurance", "tái bảo hiểm"],
        },
    },

    # ===== concept definition - English =====
    {
        "id": "en_def_surrender",
        "category": "concept_en",
        "query": "What is a surrender charge in a deferred annuity and when does it apply?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["surrender"],
            "must_cite_lesson_starting": "LOMA281_M2L3",
        },
    },
    {
        "id": "en_def_underwriting",
        "category": "concept_en",
        "query": "Explain the role of underwriting in insurance.",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["underwriting"],
        },
    },

    # ===== language override =====
    {
        "id": "lang_override_vi_to_en",
        "category": "language_override",
        "query": "Antiselection là gì? Please answer in English only.",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["antiselection"],
        },
    },
    {
        "id": "lang_override_en_to_vi",
        "category": "language_override",
        "query": "What is reinsurance? Hãy trả lời bằng tiếng Việt.",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["reinsurance", "tái bảo hiểm"],
        },
    },

    # ===== concept comparison =====
    {
        "id": "cmp_term_vs_whole_en",
        "category": "comparison_en",
        "query": "What is the difference between term life insurance and whole life insurance?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["term", "whole"],
        },
    },
    {
        "id": "cmp_speculative_vs_pure_vi",
        "category": "comparison_vi",
        "query": "So sánh speculative risk và pure risk. Loại nào có thể được bảo hiểm?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain": ["speculative", "pure"],
            "must_contain_any": ["pure risk", "rủi ro thuần"],
        },
    },
    {
        "id": "cmp_risk_transfer_vs_pooling_vi",
        "category": "comparison_vi",
        "query": "So sánh giữa Risk Transfer và Risk Pooling.",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain": ["transfer", "pooling"],
        },
    },

    # ===== KG-driven (graph traversal) =====
    {
        "id": "kg_contract_types_vi",
        "category": "kg_traversal_vi",
        # sharpened: ask for legal classifications so HyDE doesn't drift toward product types
        "query": (
            "Theo luật hợp đồng, một insurance contract có thể được phân loại "
            "thành các kiểu hợp đồng nào (ví dụ aleatory, unilateral, "
            "contract of adhesion…)? Liệt kê và mô tả ngắn gọn từng loại."
        ),
        "expects": {
            "path": "loma",
            "language": "vi",
            "min_related_nodes": 5,
            "must_contain_any": ["aleatory", "unilateral", "adhesion"],
        },
    },

    # ===== summary / overall =====
    {
        "id": "summary_module1_281_vi",
        "category": "summary_vi",
        "query": "Tóm tắt các nội dung chính của Module 1 trong LOMA 281 (Risk and Insurance).",
        # broad summarisation can legitimately fall back to web when top-5
        # chunks aren't enough to give a complete summary
        "expects": {
            "path": ["loma", "web"],
            "language": "vi",
            "must_contain_any": ["risk", "rủi ro", "insurance", "bảo hiểm"],
        },
    },
    {
        "id": "summary_annuity_overview_en",
        "category": "summary_en",
        "query": "Give me an overview of the types of annuity products covered in this course.",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["annuity"],
            "must_contain_any": ["deferred", "immediate", "fixed", "variable"],
        },
    },

    # ===== document comparison =====
    {
        "id": "doc_compare_281_vs_291_vi",
        "category": "doc_compare_vi",
        "query": "LOMA 281 và LOMA 291 khác nhau ở điểm nào về phạm vi nội dung?",
        # cross-course comparison needs broad context across both corpora;
        # accept either path
        "expects": {
            "path": ["loma", "web"],
            "language": "vi",
            "must_contain_any": ["281", "291"],
        },
    },

    # ===== web fallback (out-of-scope) =====
    {
        "id": "web_solvency_ii_vi",
        "category": "web_fallback_vi",
        "query": "Solvency II framework yêu cầu gì về capital requirements cho insurer ở EU?",
        "expects": {
            "path": "web",
            "language": "vi",
            "must_cite_pattern": r"\bWEB_\d+\b",
            "min_web_docs": 1,
        },
    },
    {
        "id": "web_insurtech_en",
        "category": "web_fallback_en",
        "query": "What are the latest insurtech innovation trends in 2024?",
        "expects": {
            "path": "web",
            "language": "en",
            "must_cite_pattern": r"\bWEB_\d+\b",
        },
    },
    {
        "id": "web_unrelated_vi",
        "category": "off_topic_vi",
        "query": "Bitcoin có phải là khoản đầu tư an toàn không?",
        # Cryptocurrency investing is outside the insurance / LOMA scope:
        # the topic classifier should short-circuit with the canned VI response.
        "expects": {
            "path": "off_topic",
            "language": "vi",
            "must_contain_any": ["Insuripedia", "kiến thức bảo hiểm"],
        },
    },

    # =========================================================================
    # ===== EXPANDED COVERAGE — added later =====
    # =========================================================================

    # ----- Course structural questions (TOC chunks) -----
    {
        "id": "struct_281_module_count_vi",
        "category": "structural_vi",
        "query": "Khóa LOMA 281 có tất cả bao nhiêu module và bao nhiêu lesson?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["4 module", "4 mô-đun"],
            "must_contain": ["14"],
        },
    },
    {
        "id": "struct_291_module_count_en",
        "category": "structural_en",
        "query": "How many modules and lessons does LOMA 291 contain in total?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["4 module"],
            "must_contain": ["12"],
        },
    },
    {
        "id": "struct_281_module4_lessons_vi",
        "category": "structural_vi",
        "query": "Module 4 của LOMA 281 có những lesson nào? Liệt kê đầy đủ.",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain": ["Group"],
            "must_contain_any": ["Group Insurance", "Group Life", "Retirement"],
        },
    },
    {
        "id": "struct_281_lesson_title_en",
        "category": "structural_en",
        "query": "What is the title of Lesson 3 in Module 1 of LOMA 281?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["Life Insurance Policies as Contracts"],
        },
    },
    {
        "id": "struct_291_module2_lessons_vi",
        "category": "structural_vi",
        "query": "Trong module 2 của LOMA 291 có bao nhiêu lesson và mỗi lesson tên gì?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain": ["4"],
            "must_contain_any": ["Distribution", "Underwriting", "Customer Service", "Claim"],
        },
    },

    # ----- Course-level metadata (Syllabus chunks) -----
    {
        "id": "syllabus_281_hours_vi",
        "category": "syllabus_vi",
        "query": "Khóa LOMA 281 dài bao nhiêu giờ?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain": ["42"],
            "must_cite_pattern": r"LOMA281_SYLLABUS",
        },
    },
    {
        "id": "syllabus_291_hours_en",
        "category": "syllabus_en",
        "query": "What is the total duration of LOMA 291 in hours?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["36"],
        },
    },
    {
        "id": "syllabus_method_vi",
        "category": "syllabus_vi",
        "query": "Phương pháp đào tạo của LOMA 281 là gì?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["self-study", "tự học"],
        },
    },
    {
        "id": "syllabus_objectives_281_en",
        "category": "syllabus_en",
        "query": "What are the main course objectives of LOMA 281?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["risk", "annuit", "insurance"],
        },
    },
    {
        "id": "syllabus_outcomes_291_vi",
        "category": "syllabus_vi",
        "query": "LOMA 291 có những kết quả đầu ra (outcomes) gì sau khi học xong?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["solvency", "stakeholder", "marketing", "tài chính", "quản"],
        },
    },
    {
        "id": "syllabus_passing_281_vi",
        "category": "syllabus_vi",
        "query": "Tiêu chí để hoàn thành (đậu) khóa LOMA 281 là gì?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["80", "quiz"],
        },
    },
    {
        "id": "syllabus_prereq_291_en",
        "category": "syllabus_en",
        "query": "What are the prerequisites for taking LOMA 291?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["none", "no prerequisite"],
        },
    },
    {
        "id": "syllabus_learners_vi",
        "category": "syllabus_vi",
        "query": "Đối tượng học của khóa LOMA 281 là ai?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["BA", "Business Analyst"],
        },
    },

    # ----- Schedule (per-module / per-lesson schedule chunks) -----
    {
        "id": "schedule_lesson_duration_vi",
        "category": "schedule_vi",
        "query": "Lesson 1 module 1 LOMA 281 mất bao nhiêu giờ và học theo cách nào?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["2", "self", "tự học", "document"],
        },
    },
    {
        "id": "schedule_module_duration_vi",
        "category": "schedule_vi",
        "query": "Học hết module 3 của LOMA 281 mất bao nhiêu giờ?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["6", "giờ", "hours"],
        },
    },
    {
        "id": "schedule_lesson_sections_vi",
        "category": "schedule_vi",
        "query": "Lesson 2 module 1 LOMA 281 có những section nào được giảng?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["Financial Institutions", "Organization", "Regulation"],
        },
    },
    {
        "id": "schedule_delivery_en",
        "category": "schedule_en",
        "query": "How is Module 2 Lesson 2 of LOMA 291 delivered?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["self-study", "document", "self study"],
        },
    },

    # ----- More concept definitions -----
    {
        "id": "concept_annuity_basics_vi",
        "category": "concept_vi",
        "query": "Annuity là gì và nó khác bảo hiểm nhân thọ ở điểm nào?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["annuity", "niên kim", "outliv", "tuổi thọ"],
        },
    },
    {
        "id": "concept_underwriting_en",
        "category": "concept_en",
        "query": "Explain the underwriting process and how underwriters classify proposed insureds into risk classes.",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["underwriting"],
            "must_contain_any": ["risk class", "preferred", "standard", "substandard"],
        },
    },
    {
        "id": "concept_group_insurance_vi",
        "category": "concept_vi",
        "query": "Group insurance khác individual insurance ở những điểm nào về cách bán và underwriting?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["group", "nhóm"],
        },
    },
    {
        "id": "concept_dividend_en",
        "category": "concept_en",
        "query": "How do policy dividends work in participating life insurance policies?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["dividend", "participating"],
        },
    },
    {
        "id": "concept_grace_period_vi",
        "category": "concept_vi",
        "query": "Grace period trong hợp đồng bảo hiểm nhân thọ có ý nghĩa gì?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["grace", "thời gian gia hạn", "premium"],
        },
    },

    # ----- More comparisons -----
    {
        "id": "compare_immediate_deferred_annuity_vi",
        "category": "comparison_vi",
        "query": "So sánh immediate annuity và deferred annuity. Khác nhau cơ bản ở điều gì?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain": ["immediate", "deferred"],
        },
    },
    {
        "id": "compare_individual_group_en",
        "category": "comparison_en",
        "query": "What are the key differences between individual life insurance and group life insurance?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain": ["individual", "group"],
        },
    },
    {
        "id": "compare_par_nonpar_en",
        "category": "comparison_en",
        "query": "Compare participating and nonparticipating life insurance policies.",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["participating", "dividend"],
        },
    },

    # ----- KG traversal English -----
    {
        "id": "kg_risk_management_en",
        "category": "kg_traversal_en",
        "query": "List the four main risk management techniques and give a short example for each.",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["avoidance", "control", "transfer", "retention", "acceptance"],
        },
    },

    # ----- Synthesis / scenario reasoning -----
    {
        "id": "synth_choose_term_life_vi",
        "category": "synthesis_vi",
        "query": "Một người trẻ độc thân, thu nhập trung bình, muốn bảo vệ tài chính cho gia đình trong 20 năm với chi phí thấp. Loại bảo hiểm nào phù hợp nhất?",
        "expects": {
            "path": "loma",
            "language": "vi",
            "must_contain_any": ["term", "tử kỳ"],
        },
    },
    {
        "id": "synth_underwriting_decision_en",
        "category": "synthesis_en",
        "query": "An applicant has high blood pressure and is overweight. How would an underwriter likely classify this applicant's risk and what factors are considered?",
        "expects": {
            "path": "loma",
            "language": "en",
            "must_contain_any": ["substandard", "risk class", "medical", "rated"],
        },
    },

    # ----- Out-of-scope / known-failure cases -----
    {
        "id": "fail_personal_advice_vi",
        "category": "fail_personal_advice",
        "query": "Tôi 35 tuổi đã có gia đình và 2 con, nên mua loại bảo hiểm cụ thể nào tốt nhất?",
        # subjective personal advice — could go either way; LOMA may give general
        # guidance, web may give purchasing tips. Just verify a reply happens.
        "expects": {
            "path": ["loma", "web"],
            "language": "vi",
        },
    },
    {
        "id": "fail_too_specific_recent",
        "category": "fail_off_corpus",
        "query": "What are the latest 2024 NAIC regulatory changes for US life insurance carriers?",
        "expects": {
            "path": "web",
            "language": "en",
            "must_cite_pattern": r"\bWEB_\d+\b",
        },
    },
    {
        "id": "fail_unrelated_vi",
        "category": "off_topic_vi",
        "query": "Cách nấu phở bò ngon nhất là gì?",
        # Cooking — clearly off-topic. Topic classifier short-circuits.
        "expects": {
            "path": "off_topic",
            "language": "vi",
            "must_contain_any": ["Insuripedia", "kiến thức bảo hiểm"],
        },
    },
    {
        "id": "fail_pure_paraphrase_vi",
        "category": "fail_paraphrase",
        # Vietnamese paraphrase of "longevity risk" without the English term —
        # tests retrieval robustness without HyDE bridging.
        "query": "Khả năng sống lâu hơn dự kiến và hết tài sản trước khi mất được gọi là gì?",
        "expects": {
            "path": ["loma", "web"],
            "language": "vi",
        },
    },

    # ----- Other-language smoke (gpt-4o handles; pipeline detector treats as 'en') -----
    {
        "id": "lang_french_concept",
        "category": "unsupported_language",
        "query": "Qu'est-ce que l'antisélection en assurance vie?",
        # French is outside SUPPORTED_LANGS = {vi, en, ja}: the gate returns
        # the English-only UNSUPPORTED_LANGUAGE_MSG verbatim, before any
        # retrieval or topic classification runs.
        "expects": {
            "path": "unsupported_language",
            "must_contain": ["Sorry", "Insuripedia", "can't understand"],
        },
    },
    {
        "id": "lang_japanese_concept",
        "category": "language_other",
        "query": "保険における逆選択（antiselection）とは何ですか？",
        "expects": {
            "path": ["loma", "web"],
        },
    },

    # ===== off-topic gate (topic classifier short-circuits before retrieval) =====
    {
        "id": "off_topic_sports_en",
        "category": "off_topic_en",
        "query": "Who won the 2024 NBA championship?",
        "expects": {
            "path": "off_topic",
            "language": "en",
            "must_contain_any": ["Insuripedia", "specializing in insurance"],
        },
    },
    {
        "id": "off_topic_cooking_en",
        "category": "off_topic_en",
        "query": "What's the best recipe for chicken parmesan?",
        "expects": {
            "path": "off_topic",
            "language": "en",
            "must_contain_any": ["Insuripedia", "cannot answer"],
        },
    },
    {
        "id": "off_topic_programming_en",
        "category": "off_topic_en",
        "query": "How do I write a Python decorator that measures function runtime?",
        "expects": {
            "path": "off_topic",
            "language": "en",
            "must_contain": ["Insuripedia"],
        },
    },
    {
        "id": "off_topic_trivia_ja",
        "category": "off_topic_ja",
        "query": "東京タワーの高さは何メートルですか？",
        "expects": {
            "path": "off_topic",
            "must_contain_any": ["Insuripedia", "保険知識"],
        },
    },

    # ===== unsupported-language gate (lang outside vi/en/ja → English canned) =====
    {
        "id": "unsupported_lang_korean",
        "category": "unsupported_language",
        "query": "보험에 대해 알려주세요",
        "expects": {
            "path": "unsupported_language",
            "must_contain": ["Sorry", "Insuripedia", "can't understand"],
        },
    },
    {
        "id": "unsupported_lang_spanish",
        "category": "unsupported_language",
        "query": "¿Qué es el seguro de vida?",
        "expects": {
            "path": "unsupported_language",
            "must_contain": ["Sorry", "Insuripedia", "can't understand"],
        },
    },
    {
        "id": "unsupported_lang_german",
        "category": "unsupported_language",
        "query": "Was ist eine Lebensversicherung?",
        "expects": {
            "path": "unsupported_language",
            "must_contain": ["Sorry", "Insuripedia", "can't understand"],
        },
    },

    # ===== quiz intent (short-circuit; no answer / no citations) =====
    {
        "id": "quiz_vi_basic",
        "category": "quiz",
        "query": "Cho tôi vài câu hỏi quiz về underwriting.",
        "expects": {"path": "quiz"},
    },
    {
        "id": "quiz_vi_module",
        "category": "quiz",
        "query": "Tạo cho tôi bài kiểm tra trắc nghiệm về Module 1 LOMA 281.",
        "expects": {"path": "quiz"},
    },
    {
        "id": "quiz_en_basic",
        "category": "quiz",
        "query": "Give me a quiz on annuities.",
        "expects": {"path": "quiz"},
    },
    {
        "id": "quiz_en_practice",
        "category": "quiz",
        "query": "Generate practice questions about LOMA 291 module 2.",
        "expects": {"path": "quiz"},
    },

    # Negative quiz cases — informational queries that mention quiz/test
    # vocabulary but should NOT be routed to the quiz intent.
    {
        "id": "quiz_neg_meta_question",
        "category": "quiz",
        "query": "Trong khóa LOMA 281 có những bài quiz nào và chúng dùng để làm gì?",
        "expects": {"path": ["loma", "web"]},
    },
    {
        "id": "quiz_neg_test_word",
        "category": "quiz",
        "query": "What does it mean when an underwriter says a case requires further testing?",
        "expects": {"path": ["loma", "web"]},
    },
]
