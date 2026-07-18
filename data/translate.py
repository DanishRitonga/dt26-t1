"""
translate.py - Chinese to English road name translation for the traffic forecasting task.

Pipeline:
  1. build_name_map()      : cn -> en for all 191 unique Chinese road names
  2. collect_text_phrases(): unique English road phrases from train + test event text
  3. coverage_check()      : how well does our translation match the actual text
  4. build_inverse_map()   : english_phrase -> [road_indices]

Outputs (in data/):
  - name_map.json          : {chinese_name: english_phrase}
  - text_phrases.txt       : all unique English road phrases found in text
  - coverage_report.txt    : which translations matched / didn't
  - name_patches.json      : manual override entries (fill in during inspection)
  - inverse_map.json       : {english_phrase: [road_indices]}
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from pypinyin import lazy_pinyin, Style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset-task1"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

ROADS_JSON = DATA / "static" / "Roads1260.json"
TRAIN_TEXT = [
    (DATA / "train" / "train_text_m1_1_11160.json", "m1"),
    (DATA / "train" / "train_text_m2_1_5039.json", "m2"),
]
TEST_TEXT = DATA / "test" / "test_texts.json"


# ---------------------------------------------------------------------------
# Translation tables
# ---------------------------------------------------------------------------
SUFFIX_DICT = {
    "路": "road",
    "桥": "bridge",
    "大街": "street",
    "高速": "expressway",
    "入口": "entrance",
    "出口": "exit",
    "辅路": "auxiliary road",
    "快速路": "expressway",
    "环": "ring",
    "二环": "second ring",
    "三环": "third ring",
    "四环": "fourth ring",
    "五环": "fifth ring",
    "北": "north",
    "南": "south",
    "东": "east",
    "西": "west",
    "中": "middle",
    "东路": "east road",
    "西路": "west road",
    "南路": "south road",
    "北路": "north road",
    "中路": "middle road",
    "内": "inner",
    "外": "outer",
}

# Expressways are translated semantically, not transliterated
EXPRESSWAY_DICT = {
    "京哈高速": "beijing harbin expressway",
    "京沪高速": "beijing shanghai expressway",
    "京台高速": "beijing taiwan expressway",
    "京港澳高速": "beijing hong kong macao expressway",
    "京藏高速": "beijing tibet expressway",
    "京新高速": "beijing urumqi expressway",
    "京承高速": "beijing chengde expressway",
    "机场高速": "airport expressway",
    "京津高速": "beijing tianjin expressway",
    "京通快速路": "beijing tongzhou expressway",
    "京开高速": "jingkai expressway",
    "京雄高速": "jingxiong expressway",
    "大兴机场高速": "daxing airport expressway",
    "机场第二高速": "airport second expressway",
}

# Special-case whole names (fill during inspection)
NAME_PATCHES: dict[str, str] = {}


def to_pinyin(han: str) -> str:
    """Transliterate Chinese chars to lowercase pinyin (no tones, no spaces)."""
    if not han:
        return ""
    return "".join(lazy_pinyin(han, style=Style.NORMAL))


def cn_to_en(name: str) -> str:
    """Translate one Chinese road name to an English phrase."""
    # 0. manual patch (highest priority)
    if name in NAME_PATCHES:
        return NAME_PATCHES[name]

    # 1. expressway full-name lookup
    for cn, en in EXPRESSWAY_DICT.items():
        if name.startswith(cn):
            rest = name[len(cn):].strip()
            if not rest:
                return en
            tail = SUFFIX_DICT.get(rest, to_pinyin(rest))
            return f"{en} {tail}".strip()

    # 2. longest structural suffix match
    for suffix_len in (4, 3, 2, 1):
        suffix = name[-suffix_len:]
        if suffix in SUFFIX_DICT:
            head = name[:-suffix_len]
            en_head = to_pinyin(head) if head else ""
            en_suffix = SUFFIX_DICT[suffix]
            return f"{en_head} {en_suffix}".strip().lower()

    # 3. fallback: pure pinyin
    return to_pinyin(name).lower()


# ---------------------------------------------------------------------------
# Step 1: build name -> english map
# ---------------------------------------------------------------------------
def load_roads() -> tuple[list[list[dict]], list[str]]:
    with open(ROADS_JSON, encoding="utf-8") as f:
        roads = json.load(f)
    # unique names, preserving first-seen order
    seen: list[str] = []
    seen_set: set[str] = set()
    for group in roads:
        for seg in group:
            n = seg.get("roadName")
            if n and n not in seen_set:
                seen.append(n)
                seen_set.add(n)
    return roads, seen


def build_name_map() -> dict[str, str]:
    """Return {chinese_name: english_phrase} for all unique names."""
    _, names = load_roads()
    name_map = {n: cn_to_en(n) for n in names}
    out = OUT / "name_map.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(name_map, f, ensure_ascii=False, indent=2)
    print(f"[build_name_map] wrote {len(name_map)} entries -> {out}")
    return name_map


# ---------------------------------------------------------------------------
# Step 2: collect unique English road phrases from event text
# ---------------------------------------------------------------------------
EVENT_PATTERN = re.compile(r"^(.+?)\s+on\s+(.+)$")


def parse_phrases(text: str) -> list[str]:
    """Extract lowercase road phrases from a single event-text string."""
    out = []
    for sentence in text.split("."):
        s = sentence.strip()
        if not s:
            continue
        m = EVENT_PATTERN.match(s)
        if m:
            out.append(m.group(2).strip().lower())
    return out


def collect_text_phrases() -> set[str]:
    """Collect all unique English road phrases across train + test text."""
    counter: Counter[str] = Counter()
    for path, _ in TRAIN_TEXT:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for v in d.values():
            counter.update(parse_phrases(v))
    with open(TEST_TEXT, encoding="utf-8") as f:
        d = json.load(f)
    for v in d.values():
        counter.update(parse_phrases(v))

    phrases = set(counter.keys())
    out_txt = OUT / "text_phrases.txt"
    out_json = OUT / "text_phrases.json"
    with open(out_txt, "w", encoding="utf-8") as f:
        for p, c in counter.most_common():
            f.write(f"{c:6d}  {p}\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dict(counter.most_common()), f, ensure_ascii=False, indent=2)
    print(f"[collect_text_phrases] {len(phrases)} unique phrases -> {out_txt}")
    return phrases


# ---------------------------------------------------------------------------
# Step 3: coverage check
# ---------------------------------------------------------------------------
def coverage_check(name_map: dict[str, str], text_phrases: set[str]) -> None:
    """
    Report how many translated names appear (exact-match or prefix) in text_phrases.
    """
    en_to_cn: dict[str, list[str]] = {}
    for cn, en in name_map.items():
        en_to_cn.setdefault(en, []).append(cn)

    matched, unmatched = [], []
    for en, cns in en_to_cn.items():
        if en in text_phrases:
            matched.append((en, cns))
        else:
            unmatched.append((en, cns))

    report = []
    report.append(f"=== COVERAGE REPORT ===")
    report.append(f"Translated names: {len(en_to_cn)}")
    report.append(f"Exact-match in text: {len(matched)}")
    report.append(f"No exact match:     {len(unmatched)}")
    report.append("")
    report.append("=== MATCHED ===")
    for en, cns in sorted(matched):
        report.append(f"  OK   {en:50s} <- {cns}")
    report.append("")
    report.append("=== UNMATCHED (candidates for manual patching) ===")
    for en, cns in sorted(unmatched):
        report.append(f"  ??   {en:50s} <- {cns}")

    out = OUT / "coverage_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[coverage_check] report -> {out}")
    print(f"  matched={len(matched)} unmatched={len(unmatched)}")


# ---------------------------------------------------------------------------
# Step 4: inverse map (english phrase -> [road_indices])
# ---------------------------------------------------------------------------
def build_inverse_map(name_map: dict[str, str]) -> dict[str, list[int]]:
    roads, _ = load_roads()
    inverse: dict[str, list[int]] = {}
    for idx, group in enumerate(roads):
        seen_in_road: set[str] = set()
        for seg in group:
            cn = seg.get("roadName")
            if not cn or cn in seen_in_road:
                continue
            seen_in_road.add(cn)
            en = name_map.get(cn)
            if en:
                inverse.setdefault(en, []).append(idx)
    out = OUT / "inverse_map.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(inverse, f, ensure_ascii=False, indent=2)
    print(f"[build_inverse_map] {len(inverse)} english phrases -> {out}")
    return inverse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # optional: load manual patches if file already exists
    patches_file = OUT / "name_patches.json"
    if patches_file.exists():
        global NAME_PATCHES
        with open(patches_file, encoding="utf-8") as f:
            NAME_PATCHES = json.load(f)
        print(f"[main] loaded {len(NAME_PATCHES)} manual patches from {patches_file}")

    name_map = build_name_map()
    text_phrases = collect_text_phrases()
    coverage_check(name_map, text_phrases)
    build_inverse_map(name_map)


if __name__ == "__main__":
    main()
