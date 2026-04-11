#!/usr/bin/env python3
"""
Regression test for the HushType AI cleanup prompt.

============================================================================
WHY THIS SCRIPT EXISTS

HushType has an opt-in "AI Cleanup" feature that runs each transcription
through a small local LLM (Qwen3-1.7B-4bit) to:
  - Convert Chinese number characters to Arabic digits in context
    (一零一大樓 → 101 大樓, but 想一下 stays as 想一下)
  - Strip leading filler words (嗯/啊/那個/...)
  - Collapse exact repetitions (我我我 → 我)

It does NOT polish, paraphrase, fix contradictions, or change word order.

The prompt that drives this is LOAD-BEARING: a 4-character punctuation
change (full-width vs half-width colons) was empirically shown to introduce
content corruption. Any future edit to the prompt MUST be re-validated
against this test suite before being committed to the production
CleanupPrompt.swift.

============================================================================
ARCHITECTURE NOTE — what this test models

Production pipeline (per integration plan v3):

    Mic → Qwen3-ASR → OpenCC s2twp → AICleaner (LLM) → TextInserter

OpenCC runs BEFORE the LLM, so the LLM only ever sees Traditional Chinese
input. This is the only script-input combination we measured as stable
(zero content corruption across 22 cases). Test cases here are all in
Traditional Chinese to match.

We deliberately do NOT run OpenCC in this script because:
  - Production runs it upstream of the LLM, not downstream
  - Inserting OpenCC after the LLM would mask drift in the LLM's output
  - We want to validate the prompt in isolation

============================================================================
HOW THIS TEST CLASSIFIES RESULTS

Each test case has an `expected` string. Results are classified as:

    ✅ PASS       — output == expected
    ⚠️  MISS      — output == input, expected != input (transform missed)
    🔴 CORRUPT    — output != expected AND output != input (changed wrong)

The PASS/MISS/CORRUPT distinction matters: MISS is benign (the LLM just
didn't apply the transformation, so the user sees their original speech),
while CORRUPT means the LLM substituted, dropped, or hallucinated content.
The exit code is non-zero if any CORRUPT cases are found.

For "must NOT change" test cases (e.g., 想一下喔), expected == input, so
any LLM modification is automatically classified as CORRUPT.

============================================================================
PROMPT MIRROR

The SYSTEM_PROMPT below MUST stay in sync with the Swift constant in
Sources/HushType/CleanupPrompt.swift. Edit one → edit the other → re-run
this test → only commit if pass count is unchanged and corrupt count is
still zero.

============================================================================
RUN

    pip3 install --system mlx-lm psutil
    python3 scripts/test_cleanup_prompt.py

Exit code 0 if no corruption, 1 if any corruption detected.
"""

import gc
import os
import sys
import time

import psutil
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"

# ============================================================================
# LOCKED PROMPT — MIRROR of Sources/HushType/CleanupPrompt.swift
# DO NOT EDIT WITHOUT RE-RUNNING THIS SCRIPT AND VERIFYING ZERO CORRUPTION.
# Even single-character changes (e.g., colon width) have been shown to
# introduce content-corruption regressions.
# ============================================================================
SYSTEM_PROMPT = """你是一個語音轉文字的後處理器。你的任務是做下面兩種**機械性**的修正。

**絕對禁止：不要改寫、不要省略內容、不要重新表達使用者的話、不要修正前後矛盾的句子（即使聽起來矛盾，那是使用者本來的話，要原封保留）、不要改變詞序、不要加標點。**

規則 1 — 中文數字轉阿拉伯數字。
**只在表達數量、編號、度量、百分比、日期、時間、小數的時候才轉換。**
- 量詞前面要轉：「五個蘋果」→「5 個蘋果」、「三本書」→「3 本書」、「二十五度」→「25 度」、「一百公里」→「100 公里」
- 編號要轉：「一零一大樓」→「101 大樓」
- 小數的「點」轉「.」：「三點一四」→「3.14」
- 百分比:「百分之三十二」→「32%」
- 年份日期：「兩千零二十六年三月五日」→「2026 年 3 月 5 日」

**「一」當助詞或固定詞時絕對不要轉**：想一下、看一看、等一等、試一試、第一、一直、一起、一定、一樣、一些、一會兒、一輩子。

規則 2 — 移除句首明顯的口頭禪贅字 + 收縮連續重複的字。
- **只移除句首**的這幾個詞：嗯、啊、呃、欸、那個、就是。例如「嗯我覺得」→「我覺得」、「那個我想問」→「我想問」。
- **不要動句子中間或結尾的這些字**——它們可能是有意義的內容。例如「我覺得這個那個其實還行」要保留「那個」不變。
- **連續重複的代名詞或語氣詞要收縮**：「我我我覺得」→「我覺得」、「然後然後」→「然後」。
- 「對對對」、「好好好」這種重複是強調，**不要收縮**。

只輸出修正後的句子，不要加前綴、不要加引號、不要解釋。繁體簡體都可以，後續會由其他工具統一處理。

範例：
輸入：我住在一零一大樓
輸出：我住在 101 大樓

輸入：我有五個蘋果
輸出：我有 5 個蘋果

輸入：想一下喔
輸出：想一下喔

輸入：今天氣溫二十五度
輸出：今天氣溫 25 度

輸入：三點一四
輸出：3.14

輸入：百分之三十二點六八
輸出：32.68%

輸入：嗯那個我覺得這個方案不錯
輸出：我覺得這個方案不錯

輸入：我我我覺得有道理
輸出：我覺得有道理

輸入：然後然後我就走了
輸出：然後我就走了

輸入：我覺得這件事其實沒那麼簡單
輸出：我覺得這件事其實沒那麼簡單"""


# (input, label, expected_output)
# All inputs are Traditional Chinese — production OpenCC runs upstream.
TEST_CASES = [
    # --- ITN — should convert numbers in context
    ("我住在一零一大樓",                    "ITN building number",     "我住在 101 大樓"),
    ("今天氣溫二十五度",                    "ITN measurement",         "今天氣溫 25 度"),
    ("三點一四",                           "ITN decimal",             "3.14"),
    ("百分之四十七點五",                    "ITN percent + decimal",   "47.5%"),
    ("買了三本書",                         "ITN with 量詞 本",        "買了 3 本書"),
    ("跑了一百公里",                       "ITN with 量詞 公里",      "跑了 100 公里"),

    # --- Particle / fixed phrase: must NOT convert 一
    ("想一下喔",                           "particle: keep",          "想一下喔"),
    ("看一看再決定",                       "particle: keep",          "看一看再決定"),
    ("我一直都在",                         "fixed phrase: keep",      "我一直都在"),
    ("那邊的一些朋友",                     "fixed phrase: keep",      "那邊的一些朋友"),

    # --- Filler removal at sentence start
    ("嗯我覺得這個方案不錯",                "strip 嗯 at start",       "我覺得這個方案不錯"),
    ("那個我想問你一個問題",                "strip 那個 at start",     "我想問你一個問題"),
    ("啊就是我想說的事情",                  "strip 啊就是 at start",   "我想說的事情"),

    # --- Repetition collapse
    ("我我我覺得有道理",                    "collapse 我我我",         "我覺得有道理"),
    ("然後然後我就走了",                    "collapse 然後然後",       "然後我就走了"),

    # --- Must NOT touch
    ("我覺得這件事其實沒那麼簡單",          "no filler — keep",        "我覺得這件事其實沒那麼簡單"),
    ("我覺得這個方案很好但是我覺得它不會成功", "contradiction — keep",   "我覺得這個方案很好但是我覺得它不會成功"),
    ("對對對沒錯",                         "emphasis — keep",         "對對對沒錯"),

    # --- Combined: filler + ITN
    ("嗯那個我有三本書",                    "filler strip + ITN",      "我有 3 本書"),

    # --- Pass-through
    ("hello world",                        "english only",            "hello world"),
]


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def mem_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def fmt_ms(t: float) -> str:
    return f"{t*1000:.0f}ms" if t < 1 else f"{t:.2f}s"


def classify(input_text: str, expected: str, actual: str) -> str:
    """Tri-state result classification."""
    if actual == expected:
        return "PASS"
    if actual == input_text:
        return "MISS"
    return "CORRUPT"


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 78)
    print(f"HushType AI Cleanup — prompt regression suite")
    print(f"Model: {MODEL_ID}")
    print("=" * 78)

    base_proc = mem_mb()
    print(f"\n[BEFORE LOAD] Process RSS: {base_proc:.0f} MB")

    print(f"\n[LOADING MODEL]")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    gc.collect()
    after_proc = mem_mb()
    print(f"  Loaded in {load_time:.1f}s — RSS now {after_proc:.0f} MB "
          f"(Δ +{after_proc - base_proc:.0f} MB)")

    sampler = make_sampler(temp=0.0)  # deterministic

    print(f"\n[RUNNING {len(TEST_CASES)} CASES]")
    print("-" * 78)

    counts = {"PASS": 0, "MISS": 0, "CORRUPT": 0}
    failures: list[tuple[int, str, str, str, str, str]] = []
    total_inference = 0.0

    for i, (text, label, expected) in enumerate(TEST_CASES, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"輸入：{text}\n輸出："},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
        )

        t0 = time.time()
        out = generate(
            model, tokenizer,
            prompt=prompt, max_tokens=128,
            sampler=sampler, verbose=False,
        )
        elapsed = time.time() - t0
        total_inference += elapsed

        actual = out.strip()
        for prefix in ("輸出：", "输出："):
            if actual.startswith(prefix):
                actual = actual[len(prefix):].strip()
                break

        result = classify(text, expected, actual)
        counts[result] += 1
        if result != "PASS":
            failures.append((i, label, text, expected, actual, result))

        marker = {"PASS": "✅", "MISS": "⚠️ ", "CORRUPT": "🔴"}[result]
        print(f"\n[{i:2d}/{len(TEST_CASES)}] {marker} {result:7s} {fmt_ms(elapsed):>6s}  ({label})")
        print(f"        in : {text}")
        if result != "PASS":
            print(f"        exp: {expected}")
        print(f"        out: {actual}")

    # ----- summary -----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Pass:    {counts['PASS']:2d} / {len(TEST_CASES)}")
    print(f"  Miss:    {counts['MISS']:2d}  (benign — transformation skipped, content unchanged)")
    print(f"  Corrupt: {counts['CORRUPT']:2d}  (LLM modified content in unexpected way)")
    print(f"  Avg inference: {total_inference/len(TEST_CASES):.2f}s per case")
    print(f"  Peak RSS:      {mem_mb():.0f} MB")

    if counts["CORRUPT"] > 0:
        print("\n🔴 CORRUPTION DETECTED — DO NOT SHIP THIS PROMPT")
        print("Failures:")
        for i, label, text, expected, actual, _ in failures:
            if _ == "CORRUPT":
                print(f"  [{i}] {label}")
                print(f"      in : {text}")
                print(f"      exp: {expected}")
                print(f"      out: {actual}")
        return 1

    print("\n✅ Zero corruption — prompt is safe to ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
