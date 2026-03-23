#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local analyzer ported from https://mjanalyzer.netlify.app/.
No browser is required.

Main entry points:
- analyze_tiles(hand_tiles, dead_tiles)
- analyze_round_state(round_state, states)
- annotate_states_shanten(states)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import random
import re

from fileProcess import RoundState, States

def parse_list(cards: List[str]) -> str:
    type_dict = {0: '', 1: 'm', 2: 'p', 3: 's', 4: 'z'}
    out = ""
    for card in cards:
        card_num = int(card)
        out += f"{int(card_num / 10 % 10)}{type_dict[int(card_num / 100)]}"
    return out

SUITS = ("m", "p", "s", "z")
ALL_TILE_IDS: List[str] = []
for _suit in SUITS:
    _max_rank = 7 if _suit == "z" else 9
    for _rank in range(1, _max_rank + 1):
        ALL_TILE_IDS.append(f"{_rank}{_suit}")

TILE_TO_INDEX = {tile: i for i, tile in enumerate(ALL_TILE_IDS)}
TILE_KIND_COUNT = len(ALL_TILE_IDS)  # 34
TOTAL_TILE_COUNT = 136
MAX_TILE_COPY = 4
MAX_HAND_SIZE = 17

HONOR_LABELS = ["東", "南", "西", "北", "白", "發", "中"]
SUIT_TEXT = {"m": "萬", "p": "餅", "s": "索"}

_COMPACT_TOKEN_RE = re.compile(r"(\d+)([mpsz])")


def tile_to_label(tile_id: str) -> str:
    suit = tile_id[1]
    value = int(tile_id[0])
    if suit == "z":
        return HONOR_LABELS[value - 1]
    return f"{value}{SUIT_TEXT[suit]}"


def _validate_tile_id(tile_id: str) -> None:
    if len(tile_id) != 2:
        raise ValueError(f"Invalid tile id: {tile_id}")
    value, suit = tile_id[0], tile_id[1]
    if suit not in SUITS:
        raise ValueError(f"Invalid tile suit: {tile_id}")
    if not value.isdigit():
        raise ValueError(f"Invalid tile value: {tile_id}")
    num = int(value)
    if suit == "z":
        if num < 1 or num > 7:
            raise ValueError(f"Honor tile out of range: {tile_id}")
    else:
        if num < 1 or num > 9:
            raise ValueError(f"Suit tile out of range: {tile_id}")


def parse_compact_tiles(text: str) -> List[str]:
    s = (text or "").strip().lower()
    if not s:
        return []
    matches = list(_COMPACT_TOKEN_RE.finditer(s))
    if not matches:
        raise ValueError(f"Invalid tile string: {text}")
    joined = "".join(m.group(0) for m in matches)
    if joined != s:
        raise ValueError(f"Invalid tile string: {text}")

    out: List[str] = []
    for m in matches:
        digits, suit = m.group(1), m.group(2)
        for ch in digits:
            tile_id = f"{ch}{suit}"
            _validate_tile_id(tile_id)
            out.append(tile_id)
    return out


def state_cards_to_tile_ids(cards: Sequence[str]) -> List[str]:
    # Reuse analyze.py conversion style: 211 -> 1p
    return parse_compact_tiles(parse_list(list(cards)))


def _normalize_dead_counts(
    dead_tiles: Optional[Iterable[str] | Mapping[str, int]],
) -> Dict[str, int]:
    if dead_tiles is None:
        return {}
    if isinstance(dead_tiles, Mapping):
        out: Dict[str, int] = {}
        for tile_id, count in dead_tiles.items():
            if count <= 0:
                continue
            _validate_tile_id(tile_id)
            out[tile_id] = int(count)
        return out

    out_counter: Counter[str] = Counter()
    for tile_id in dead_tiles:
        _validate_tile_id(tile_id)
        out_counter[tile_id] += 1
    return dict(out_counter)


def validate_counts(hand_tiles: Sequence[str], dead_counts: Mapping[str, int]) -> None:
    if not hand_tiles:
        raise ValueError("手牌不可為空")
    if len(hand_tiles) % 3 == 0:
        raise ValueError("牌張數不可為 3 的倍數 (例如 3, 6, 9, 12, 15)")
    if len(hand_tiles) > MAX_HAND_SIZE:
        raise ValueError("手牌張數不可超過 17")

    hand_counter = Counter(hand_tiles)
    for tile_id, count in hand_counter.items():
        _validate_tile_id(tile_id)
        if count > MAX_TILE_COPY:
            raise ValueError(f"手牌 {tile_id} 超過 4 張")

    for tile_id, count in dead_counts.items():
        _validate_tile_id(tile_id)
        if count > MAX_TILE_COPY:
            raise ValueError(f"已現出牌 {tile_id} 超過 4 張")

    for tile_id in set(hand_counter) | set(dead_counts):
        if hand_counter[tile_id] + dead_counts.get(tile_id, 0) > MAX_TILE_COPY:
            raise ValueError(f"{tile_id} 手牌+已現出超過 4 張")


@dataclass
class TileStat:
    id: str
    remaining: int


@dataclass
class EffectiveStats:
    totalRemaining: int
    probability: float
    tileStats: List[TileStat] = field(default_factory=list)


@dataclass
class Recommendation:
    discardTile: str
    explanation: str
    effectiveTiles: List[str] = field(default_factory=list)
    effectiveStats: Optional[EffectiveStats] = None
    drawStats: Optional[EffectiveStats] = None


@dataclass
class MeldRecommendation:
    actionType: str
    triggerTile: str
    meldTiles: List[str]
    discardTile: str
    shantenAfter: int
    effectiveTiles: List[str]
    totalRemaining: int
    effectiveStats: Optional[EffectiveStats] = None


@dataclass
class SummaryStats:
    totalEffective: int
    probability: float


@dataclass
class LocalAnalysisResult:
    shanten: int
    statusText: str
    generalAdvice: str
    difficultyScore: int
    difficultyGrade: str
    recommendations: List[Recommendation] = field(default_factory=list)
    pongRecommendations: List[MeldRecommendation] = field(default_factory=list)
    chiRecommendations: List[MeldRecommendation] = field(default_factory=list)
    kongRecommendations: List[MeldRecommendation] = field(default_factory=list)
    summaryStats: SummaryStats = field(default_factory=lambda: SummaryStats(0, 0.0))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ShantenCalculator:
    # Port of site class Ke
    def __init__(self, hand_size: int):
        self.target_sets = hand_size // 3

    def calculate(self, hand_tiles: Sequence[str]) -> int:
        counts = [0] * TILE_KIND_COUNT
        for tile_id in hand_tiles:
            idx = TILE_TO_INDEX.get(tile_id)
            if idx is not None:
                counts[idx] += 1
        return self._get_shanten(counts)

    def _get_shanten(self, counts: List[int]) -> int:
        best = 99

        # Pair candidate branch
        for i in range(TILE_KIND_COUNT):
            if counts[i] >= 2:
                counts[i] -= 2
                score = self._calculate_mentsu_tatsu(counts)
                shanten = 2 * self.target_sets - score - 1
                counts[i] += 2
                if shanten < best:
                    best = shanten

        # No pair branch
        score = self._calculate_mentsu_tatsu(counts)
        shanten = 2 * self.target_sets - score
        if shanten < best:
            best = shanten
        return best

    def _calculate_mentsu_tatsu(self, counts: List[int]) -> int:
        return self._search_groups(counts[:], 0, 0)

    def _search_groups(self, counts: List[int], start: int, sets_count: int) -> int:
        while start < TILE_KIND_COUNT and counts[start] == 0:
            start += 1
        if start >= TILE_KIND_COUNT:
            return self._calculate_tatsu(counts, sets_count)

        best = -1

        # Triplet
        if counts[start] >= 3:
            counts[start] -= 3
            value = self._search_groups(counts, start, sets_count + 1)
            if value > best:
                best = value
            counts[start] += 3

        # Sequence (only numbered tiles)
        if start < 27:
            suit = start // 9
            if start % 9 <= 6 and counts[start + 1] > 0 and counts[start + 2] > 0:
                suit1 = (start + 1) // 9
                suit2 = (start + 2) // 9
                if suit1 == suit and suit2 == suit:
                    counts[start] -= 1
                    counts[start + 1] -= 1
                    counts[start + 2] -= 1
                    value = self._search_groups(counts, start, sets_count + 1)
                    if value > best:
                        best = value
                    counts[start] += 1
                    counts[start + 1] += 1
                    counts[start + 2] += 1

        # Skip current tile
        value = self._search_groups(counts, start + 1, sets_count)
        if value > best:
            best = value
        return best

    def _calculate_tatsu(self, counts: List[int], sets_count: int) -> int:
        tatsu = 0
        work = counts[:]

        for i in range(TILE_KIND_COUNT):
            while work[i] > 0:
                if i < 27:
                    offset = i % 9
                    suit = i // 9
                    if offset <= 7 and work[i + 1] > 0 and (i + 1) // 9 == suit:
                        work[i] -= 1
                        work[i + 1] -= 1
                        tatsu += 1
                        continue
                    if offset <= 6 and work[i + 2] > 0 and (i + 2) // 9 == suit:
                        work[i] -= 1
                        work[i + 2] -= 1
                        tatsu += 1
                        continue

                if work[i] >= 2:
                    work[i] -= 2
                    tatsu += 1
                    continue

                work[i] -= 1

        capped_tatsu = tatsu
        if sets_count + tatsu > self.target_sets:
            capped_tatsu = max(0, self.target_sets - sets_count)
        return 2 * sets_count + capped_tatsu


def _base_shanten(hand_tiles: Sequence[str]) -> int:
    return ShantenCalculator(len(hand_tiles)).calculate(hand_tiles)


def _best_discard_after_draw(
    hand_tiles: Sequence[str],
    dead_counts: Mapping[str, int],
    calculator: ShantenCalculator,
) -> Tuple[int, str, List[str], int]:
    # Port of site function Tf
    best_discard = ""
    best_shanten = 99
    best_total_remaining = -1
    best_effective_tiles: List[str] = []

    unique_tiles = list(dict.fromkeys(hand_tiles))

    for tile in unique_tiles:
        tmp = list(hand_tiles)
        tmp.remove(tile)
        shanten = calculator.calculate(tmp)
        if shanten < best_shanten:
            best_shanten = shanten

    for tile in unique_tiles:
        tmp = list(hand_tiles)
        tmp.remove(tile)
        shanten = calculator.calculate(tmp)
        if shanten != best_shanten:
            continue

        total_remaining = 0
        effective_tiles: List[str] = []
        for draw in ALL_TILE_IDS:
            test = tmp + [draw]
            next_shanten = calculator.calculate(test)
            improves = next_shanten == -1 if shanten == -1 else next_shanten < shanten
            if not improves:
                continue

            effective_tiles.append(draw)
            in_hand = tmp.count(draw)
            seen = dead_counts.get(draw, 0)
            total_remaining += max(0, MAX_TILE_COPY - in_hand - seen)

        if total_remaining > best_total_remaining:
            best_total_remaining = total_remaining
            best_discard = tile
            best_effective_tiles = effective_tiles

    return best_shanten, best_discard, best_effective_tiles, best_total_remaining


def _effective_stats(
    tile_ids: Sequence[str],
    hand_tiles: Sequence[str],
    dead_counts: Mapping[str, int],
) -> EffectiveStats:
    denominator = max(1, TOTAL_TILE_COUNT - len(hand_tiles) - sum(dead_counts.values()))
    total_remaining = 0
    tile_stats: List[TileStat] = []

    for tile in tile_ids:
        in_hand = hand_tiles.count(tile)
        seen = dead_counts.get(tile, 0)
        remaining = max(0, MAX_TILE_COPY - in_hand - seen)
        total_remaining += remaining
        tile_stats.append(TileStat(id=tile, remaining=remaining))

    probability = (total_remaining / denominator) * 100
    return EffectiveStats(
        totalRemaining=total_remaining,
        probability=probability,
        tileStats=tile_stats,
    )


def _status_text(shanten: int, is_discard_phase: bool) -> str:
    if shanten == -1:
        return "胡牌"
    if is_discard_phase:
        if shanten == 0:
            return "捨一張即聽牌"
        if shanten == 1:
            return "捨一張後1進聽"
        if shanten == 2:
            return "捨一張後2進聽"
        return f"捨一張後{shanten}進聽"
    if shanten == 0:
        return "已聽牌"
    if shanten == 1:
        return "1進聽"
    if shanten == 2:
        return "2進聽"
    return f"{shanten}進聽"


def _difficulty_score(shanten: int, total_effective: int) -> Tuple[int, str]:
    if shanten == -1:
        score = 100
    else:
        base = 0
        if shanten == 0:
            base = 82
        elif shanten == 1:
            base = 66
        elif shanten == 2:
            base = 50
        elif shanten == 3:
            base = 40
        elif shanten == 4:
            base = 30
        elif shanten == 5:
            base = 20
        elif shanten == 6:
            base = 10
        elif shanten == 7:
            base = 5

        bonus = 0.0
        cap = 0.0
        if shanten == 0:
            bonus, cap = total_effective * 1.5, 17
        elif shanten == 1:
            bonus, cap = total_effective * 0.8, 24
        elif shanten == 2:
            bonus, cap = total_effective * 0.6, 25
        elif shanten == 3:
            bonus, cap = total_effective * 0.4, 20
        elif shanten == 4:
            bonus, cap = total_effective * 0.2, 10
        bonus = min(bonus, cap)
        score = int(round(min(99, max(0, base + bonus))))

    if score >= 90:
        grade = "S"
    elif score >= 70:
        grade = "A"
    elif score >= 50:
        grade = "B"
    elif score >= 30:
        grade = "C"
    elif score > 0:
        grade = "D"
    else:
        grade = "E"
    return score, grade


def analyze_tiles(
    hand_tiles: Sequence[str],
    dead_tiles: Optional[Iterable[str] | Mapping[str, int]] = None,
    validate: bool = True,
) -> LocalAnalysisResult:
    """
    Port of site function Nh.
    Input hand tiles/dead tiles use tile ids (e.g. ['1m', '2m', '3m']).
    """
    hand = list(hand_tiles)
    dead_counts = _normalize_dead_counts(dead_tiles)
    if validate:
        validate_counts(hand, dead_counts)

    hand_size = len(hand)
    is_discard_phase = hand_size % 3 == 2
    calculator = ShantenCalculator(hand_size)
    shanten = calculator.calculate(hand)

    dead_total = sum(dead_counts.values())
    denominator = max(1, TOTAL_TILE_COUNT - hand_size - dead_total)

    recommendations: List[Recommendation] = []
    pong_recs: List[MeldRecommendation] = []
    chi_recs: List[MeldRecommendation] = []
    kong_recs: List[MeldRecommendation] = []
    total_effective = 0

    if is_discard_phase:
        # Recommend discard
        candidates: List[Tuple[str, List[str], int]] = []
        for discard in dict.fromkeys(hand):
            tmp = hand[:]
            tmp.remove(discard)

            effective_tiles: List[str] = []
            total_remaining = 0
            for draw in ALL_TILE_IDS:
                test = tmp + [draw]
                next_shanten = calculator.calculate(test)
                improves = next_shanten == -1 if shanten == -1 else next_shanten < shanten
                if not improves:
                    continue

                effective_tiles.append(draw)
                in_hand = tmp.count(draw)
                seen = dead_counts.get(draw, 0)
                total_remaining += max(0, MAX_TILE_COPY - in_hand - seen)

            if effective_tiles:
                candidates.append((discard, effective_tiles, total_remaining))

        candidates.sort(key=lambda x: x[2], reverse=True)
        if candidates:
            total_effective = candidates[0][2]
            for discard, effective_tiles, total_remaining in candidates:
                explanation = (
                    f"打 {tile_to_label(discard)}。"
                    f"進張 {total_remaining} 張 ({len(effective_tiles)} 種)。"
                )
                recommendations.append(
                    Recommendation(
                        discardTile=discard,
                        explanation=explanation,
                        effectiveTiles=effective_tiles,
                    )
                )

        # Recommend kong when 4-of-a-kind already in hand
        hand_counter = Counter(hand)
        for tile_id, count in hand_counter.items():
            if count != 4:
                continue

            tmp = [x for x in hand if x != tile_id]
            tmp_calc = ShantenCalculator(len(tmp))
            shanten_after = tmp_calc.calculate(tmp)

            dead_after = dict(dead_counts)
            dead_after[tile_id] = dead_after.get(tile_id, 0) + 4

            effective_tiles: List[str] = []
            total_remaining = 0
            for draw in ALL_TILE_IDS:
                test = tmp + [draw]
                next_shanten = tmp_calc.calculate(test)
                if next_shanten < shanten_after or (shanten_after == -1 and next_shanten == -1):
                    effective_tiles.append(draw)
                    in_hand = tmp.count(draw)
                    seen = dead_after.get(draw, 0)
                    total_remaining += max(0, MAX_TILE_COPY - in_hand - seen)

            kong_recs.append(
                MeldRecommendation(
                    actionType="kong",
                    triggerTile=tile_id,
                    meldTiles=[tile_id, tile_id, tile_id, tile_id],
                    discardTile="",
                    shantenAfter=shanten_after,
                    effectiveTiles=effective_tiles,
                    totalRemaining=total_remaining,
                )
            )

        kong_recs.sort(key=lambda x: (x.shantenAfter, -x.totalRemaining))

    else:
        # Recommend draw
        draw_candidates: List[Tuple[str, str, List[str], int, bool]] = []
        total_effective_count = 0
        hand_counter = Counter(hand)

        for draw in ALL_TILE_IDS:
            test_hand = hand + [draw]
            next_shanten = calculator.calculate(test_hand)
            if next_shanten >= shanten:
                continue

            seen = dead_counts.get(draw, 0)
            in_hand = hand_counter.get(draw, 0)
            remaining = max(0, MAX_TILE_COPY - in_hand - seen)
            if remaining == 0:
                continue

            total_effective_count += remaining
            if next_shanten == -1:
                draw_candidates.append((draw, "", [], remaining, True))
            else:
                next_calc = ShantenCalculator(len(test_hand))
                _, discard, waits, _ = _best_discard_after_draw(
                    test_hand,
                    dead_counts,
                    next_calc,
                )
                draw_candidates.append((draw, discard, waits, remaining, False))

        total_effective = total_effective_count
        draw_candidates.sort(key=lambda x: x[3], reverse=True)
        for draw, discard, waits, _remaining, is_win in draw_candidates:
            if is_win:
                explanation = f"進 {tile_to_label(draw)} 即自摸胡牌！"
                recommendations.append(
                    Recommendation(
                        discardTile=draw,
                        explanation=explanation,
                        effectiveTiles=[],
                    )
                )
            else:
                explanation = f"若進 {tile_to_label(draw)}，建議捨 {tile_to_label(discard)}。"
                recommendations.append(
                    Recommendation(
                        discardTile=draw,
                        explanation=explanation,
                        effectiveTiles=waits,
                    )
                )

        # Pong / kong recommendations (if other player discards)
        for trigger in dict.fromkeys(hand):
            count = hand_counter[trigger]

            if count >= 2:
                tmp = hand[:]
                tmp.remove(trigger)
                tmp.remove(trigger)

                tmp_calc = ShantenCalculator(len(tmp))
                dead_after = dict(dead_counts)
                dead_after[trigger] = dead_after.get(trigger, 0) + 1
                shanten_after, discard, effective, total_remaining = _best_discard_after_draw(
                    tmp,
                    dead_after,
                    tmp_calc,
                )
                pong_recs.append(
                    MeldRecommendation(
                        actionType="pong",
                        triggerTile=trigger,
                        meldTiles=[trigger, trigger],
                        discardTile=discard,
                        shantenAfter=shanten_after,
                        effectiveTiles=effective,
                        totalRemaining=total_remaining,
                    )
                )

            if count >= 3:
                tmp = hand[:]
                for _ in range(3):
                    tmp.remove(trigger)

                tmp_calc = ShantenCalculator(len(tmp))
                shanten_after = tmp_calc.calculate(tmp)

                dead_after = dict(dead_counts)
                dead_after[trigger] = dead_after.get(trigger, 0) + 1

                effective_tiles: List[str] = []
                total_remaining = 0
                for draw in ALL_TILE_IDS:
                    test = tmp + [draw]
                    next_shanten = tmp_calc.calculate(test)
                    if next_shanten < shanten_after or (shanten_after == -1 and next_shanten == -1):
                        effective_tiles.append(draw)
                        in_hand = tmp.count(draw)
                        seen = dead_after.get(draw, 0)
                        total_remaining += max(0, MAX_TILE_COPY - in_hand - seen)

                kong_recs.append(
                    MeldRecommendation(
                        actionType="kong",
                        triggerTile=trigger,
                        meldTiles=[trigger, trigger, trigger, trigger],
                        discardTile="",
                        shantenAfter=shanten_after,
                        effectiveTiles=effective_tiles,
                        totalRemaining=total_remaining,
                    )
                )

        pong_recs.sort(key=lambda x: (x.shantenAfter, -x.totalRemaining))
        kong_recs.sort(key=lambda x: (x.shantenAfter, -x.totalRemaining))

        # Chi recommendations (if upper player discards)
        for trigger in ALL_TILE_IDS:
            if trigger.endswith("z"):
                continue

            suit = trigger[1]
            value = int(trigger[0])
            combos: List[List[str]] = []

            p1 = f"{value + 1}{suit}"
            p2 = f"{value + 2}{suit}"
            m1 = f"{value - 1}{suit}"
            m2 = f"{value - 2}{suit}"

            if hand_counter.get(p1, 0) > 0 and hand_counter.get(p2, 0) > 0:
                combos.append([p1, p2])
            if hand_counter.get(m1, 0) > 0 and hand_counter.get(p1, 0) > 0:
                combos.append([m1, p1])
            if hand_counter.get(m2, 0) > 0 and hand_counter.get(m1, 0) > 0:
                combos.append([m2, m1])

            for meld in combos:
                tmp = hand[:]
                for tile in meld:
                    tmp.remove(tile)

                tmp_calc = ShantenCalculator(len(tmp))
                dead_after = dict(dead_counts)
                dead_after[trigger] = dead_after.get(trigger, 0) + 1
                shanten_after, discard, effective, total_remaining = _best_discard_after_draw(
                    tmp,
                    dead_after,
                    tmp_calc,
                )
                chi_recs.append(
                    MeldRecommendation(
                        actionType="chi",
                        triggerTile=trigger,
                        meldTiles=meld,
                        discardTile=discard,
                        shantenAfter=shanten_after,
                        effectiveTiles=effective,
                        totalRemaining=total_remaining,
                    )
                )

        chi_recs.sort(key=lambda x: (x.shantenAfter, -x.totalRemaining))

    # Match site UI's effective tile statistics formula (tl)
    for rec in recommendations:
        rec.effectiveStats = _effective_stats(rec.effectiveTiles, hand, dead_counts)
        if not is_discard_phase:
            rec.drawStats = _effective_stats([rec.discardTile], hand, dead_counts)

    for rec in pong_recs:
        rec.effectiveStats = _effective_stats(rec.effectiveTiles, hand, dead_counts)
    for rec in chi_recs:
        rec.effectiveStats = _effective_stats(rec.effectiveTiles, hand, dead_counts)
    for rec in kong_recs:
        rec.effectiveStats = _effective_stats(rec.effectiveTiles, hand, dead_counts)

    status_text = _status_text(shanten, is_discard_phase)
    difficulty_score, difficulty_grade = _difficulty_score(shanten, total_effective)
    summary = SummaryStats(
        totalEffective=total_effective,
        probability=(total_effective / denominator) * 100,
    )

    return LocalAnalysisResult(
        shanten=shanten,
        statusText=status_text,
        generalAdvice="",
        difficultyScore=difficulty_score,
        difficultyGrade=difficulty_grade,
        recommendations=recommendations,
        pongRecommendations=pong_recs,
        chiRecommendations=chi_recs,
        kongRecommendations=kong_recs,
        summaryStats=summary,
    )


def analyze_round_state(
    ref: RoundState,
    states: States,
    include_winner_view: bool = True,
    validate: bool = False,
) -> Dict[str, LocalAnalysisResult]:
    """
    Analyze one RoundState directly from fileProcess.py data structure.
    Returns keys: winner (optional), E, S, W, N.
    """
    dead_tiles = state_cards_to_tile_ids(ref.abandonTiles)
    dead_counts = Counter(dead_tiles)

    out: Dict[str, LocalAnalysisResult] = {}
    if include_winner_view:
        winner = states.winnerLoc or "E"
        winner_tiles = state_cards_to_tile_ids(states.get_player(ref, winner).tiles)
        out["winner"] = analyze_tiles(winner_tiles, dead_counts, validate=validate)

    for loc in ("E", "S", "W", "N"):
        hand_tiles = state_cards_to_tile_ids(states.get_player(ref, loc).tiles)
        out[loc] = analyze_tiles(hand_tiles, dead_counts, validate=validate)
    return out


def calculate_shanten(
    hand_tiles: Sequence[str],
    dead_tiles: Optional[Iterable[str] | Mapping[str, int]] = None,
    validate: bool = True,
) -> int:
    dead_counts = _normalize_dead_counts(dead_tiles)
    if validate:
        validate_counts(hand_tiles, dead_counts)
    return ShantenCalculator(len(hand_tiles)).calculate(hand_tiles)


def annotate_states_shanten(states: States, validate: bool = False) -> None:
    """
    Fill each step's 4-player shantenCount in-place:
    ref.player[idx].shantenCount = local analyzed shanten
    """
    for ref in states.state:
        dead_tiles = state_cards_to_tile_ids(ref.abandonTiles)
        dead_counts = Counter(dead_tiles)
        for loc in ("E", "S", "W", "N"):
            idx = states.player_index_from_loc(loc)
            if idx < 0 or idx >= 4:
                raise ValueError("states.playerBank 未設定，無法定位玩家座位")
            hand_tiles = state_cards_to_tile_ids(states.get_player(ref, loc).tiles)
            ref.player[idx].shantenCount = calculate_shanten(
                hand_tiles,
                dead_counts,
                validate=validate,
            )


def random_hand(hand_size: int, rng: Optional[random.Random] = None) -> List[str]:
    rng = rng or random.Random()
    deck: List[str] = []
    for tile_id in ALL_TILE_IDS:
        deck.extend([tile_id] * MAX_TILE_COPY)
    rng.shuffle(deck)
    return deck[:hand_size]


def _sim_shanten_for_3n2_hand(hand_tiles: Sequence[str]) -> int:
    # Port of site function zh
    hand = list(hand_tiles)
    base = _base_shanten(hand)
    if len(hand) % 3 == 2:
        if base == -1:
            return -1
        best = 99
        visited = set()
        for i, tile in enumerate(hand):
            if tile in visited:
                continue
            visited.add(tile)
            tmp = hand[:i] + hand[i + 1 :]
            value = _base_shanten(tmp)
            if value < best:
                best = value
                if best == 0:
                    break
        return best
    return base


def simulate_after_rounds(
    hand_tiles: Sequence[str],
    rounds: int,
    rng: Optional[random.Random] = None,
) -> int:
    # Port of site function Ah
    rng = rng or random.Random()
    hand = list(hand_tiles)

    in_hand = Counter(hand)
    wall: List[str] = []
    for tile in ALL_TILE_IDS:
        for _ in range(MAX_TILE_COPY - in_hand[tile]):
            wall.append(tile)
    wall_len = len(wall)

    def draw_tile() -> Optional[str]:
        nonlocal wall_len
        if wall_len <= 0:
            return None
        idx = rng.randrange(wall_len)
        tile = wall[idx]
        wall[idx] = wall[wall_len - 1]
        wall_len -= 1
        return tile

    def discard_best(tmp_hand: List[str]) -> None:
        best = 99
        candidate_idx: List[int] = []
        visited = set()
        for i, tile in enumerate(tmp_hand):
            if tile in visited:
                continue
            visited.add(tile)
            reduced = tmp_hand[:i] + tmp_hand[i + 1 :]
            value = _base_shanten(reduced)
            if value < best:
                best = value
                candidate_idx = [i]
            elif value == best:
                candidate_idx.append(i)
        if candidate_idx:
            remove_idx = rng.choice(candidate_idx)
            del tmp_hand[remove_idx]

    for _ in range(rounds):
        if len(hand) % 3 == 1:
            draw = draw_tile()
            if draw is None:
                break
            hand.append(draw)
        if _base_shanten(hand) == -1:
            return -1
        discard_best(hand)

    return _base_shanten(hand)


def simulate_distribution(
    hand_size: int,
    rounds: int,
    iterations: int,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Port of website simulator summary for local use.
    """
    rng = random.Random(seed)
    distribution: Dict[int, int] = {}
    shanten_sum = 0
    shanten_count = 0

    for _ in range(iterations):
        hand = random_hand(hand_size, rng=rng)
        if rounds == 0:
            if hand_size % 3 == 2:
                value = _base_shanten(hand)
                value = -1 if value == -1 else 99
            else:
                value = _sim_shanten_for_3n2_hand(hand)
        else:
            value = simulate_after_rounds(hand, rounds, rng=rng)

        distribution[value] = distribution.get(value, 0) + 1
        if value != 99:
            shanten_sum += value
            shanten_count += 1

    average_shanten = (shanten_sum / shanten_count) if shanten_count > 0 else 0.0
    return {
        "distribution": distribution,
        "averageShanten": average_shanten,
        "total": iterations,
    }
