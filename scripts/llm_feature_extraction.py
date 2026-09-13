"""LLM-based feature engineering for the VIHSD hate-speech classifier.

Uses the Claude API (Anthropic) to extract structured pragmatic signals per
comment (targeting, explicit threat/dehumanizing language, severity, and an
independent LLM-suggested label) that a pure text classifier tends to miss —
aimed specifically at the OFFENSIVE-vs-HATE boundary, which the project's own
error analysis (see docs/research_notes.md) identified as the dominant
confusion (these two classes differ mainly by *target* and *intent*, not
surface wording).

This is a feature-engineering aid, not a replacement classifier: outputs get
combined with the existing model's predictions by a downstream script
(see combine_llm_features.py), not used as the final label directly.

Usage:
    export ANTHROPIC_API_KEY=...
    python NLP/scripts/llm_feature_extraction.py --split dev
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent
DATA_DIR = FINAL_DIR / "data"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def model_slug(model: str) -> str:
    """Filename tag for a given model id -- empty for the original default (Haiku) so the
    existing `llm_features_{split}.jsonl` cache from the first experiment stays untouched,
    non-default models (e.g. a Sonnet re-run) get their own `llm_features_{slug}_{split}.jsonl`."""
    if model == DEFAULT_MODEL:
        return ""
    for name in ("haiku", "sonnet", "opus"):
        if name in model:
            return name
    return model.replace("/", "_").replace(":", "_")

SYSTEM_PROMPT = """Ban la 1 cong cu trich xuat dac trung (feature extractor) cho bai toan phan loai \
binh luan mang xa hoi tieng Viet thanh CLEAN / OFFENSIVE / HATE. Nhiem vu CUA BAN KHONG PHAI la phan loai \
cuoi cung (da co model rieng lam viec do) - ban chi trich xuat cac tin hieu ngu dung co cau truc ma \
1 classifier text thuan kho bat duoc, dac biet la ranh gioi giua OFFENSIVE (xuc pham chung chung) va \
HATE (nham vao 1 ca nhan/nhom cu the voi y dinh de doa/ha thap nhan pham).

Tra loi CHI 1 doi tuong JSON duy nhat, khong them text nao khac, dung dinh dang:
{"targets_individual_or_group": true|false, "explicit_threat_or_dehumanizing": true|false, \
"severity": 0|1|2|3, "llm_suggested_label": "CLEAN"|"OFFENSIVE"|"HATE", "rationale": "<toi da 20 tu>"}

- targets_individual_or_group: co nham cu the vao 1 ca nhan/nhom (dan toc, gioi tinh, vung mien, ton giao...) khong.
- explicit_threat_or_dehumanizing: co ngon ngu de doa ro rang hoac phi nhan hoa (vd goi la "sinh vat", "rac", doi giet...) khong.
- severity: 0=khong co gi sai, 1=kho chiu nhe, 2=xuc pham ro rang, 3=thu ghet/de doa nghiem trong.
- llm_suggested_label: nhan ban de xuat doc lap, dua tren dinh nghia CLEAN/OFFENSIVE/HATE chuan cua tap VIHSD.
- rationale: ly do ngan gon, tieng Viet, khong dau cung duoc, toi da 20 tu."""


def build_sample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """All OFFENSIVE(1)+HATE(2) rows (rare, hard boundary) + ~1.25x random CLEAN(0)
    for cost/time control — prioritizes exactly the rows this feature is meant to help."""
    rare = df[df["label_id"].isin([1, 2])]
    clean_pool = df[df["label_id"] == 0]
    n_clean = min(len(clean_pool), int(len(rare) * 1.25))
    clean_sample = clean_pool.sample(n=n_clean, random_state=seed)
    sample = pd.concat([rare, clean_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sample


def call_claude(client, model: str, text: str, max_retries: int = 4) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f'Binh luan: "{text}"'}],
            )
            raw = resp.content[0].text.strip()
            # models occasionally wrap JSON in ```json fences despite instructions
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw[raw.find("{"): raw.rfind("}") + 1]
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if attempt == max_retries - 1:
                return {"error": f"json_decode_failed: {exc}", "raw": raw}
        except Exception as exc:  # rate limit / API errors
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {type(exc).__name__}: {exc} — waiting {wait}s")
            time.sleep(wait)
    return {"error": "max_retries_exceeded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id to use")
    parser.add_argument("--limit", type=int, default=None, help="cap number of rows (for a quick smoke test)")
    args = parser.parse_args()

    import anthropic

    # .strip(): the env var occasionally carries trailing whitespace/newline (e.g. from how it
    # was set), which httpx rejects outright as an "Illegal header value" - strip defensively.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())

    df = pd.read_csv(DATA_DIR / f"{args.split}.csv")
    sample = build_sample(df)
    if args.limit:
        sample = sample.head(args.limit)

    slug = model_slug(args.model)
    out_path = DATA_DIR / f"llm_features_{slug + '_' if slug else ''}{args.split}.jsonl"
    already_done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["row_index"])
                except Exception:
                    continue
        print(f"Resuming: {len(already_done)} rows already cached in {out_path}")

    print(f"Sample size: {len(sample)} (label dist: {dict(sample['label_id'].value_counts())})")
    print(f"Model: {args.model}")

    n_calls = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for idx, row in sample.iterrows():
            if idx in already_done:
                continue
            result = call_claude(client, args.model, row["free_text"])
            record = {
                "row_index": int(idx),
                "free_text": row["free_text"],
                "true_label_id": int(row["label_id"]),
                **result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_calls += 1
            if n_calls % 25 == 0:
                print(f"  ...{n_calls} calls done")

    print(f"Done. {n_calls} new calls made this run. Output: {out_path}")


if __name__ == "__main__":
    main()
