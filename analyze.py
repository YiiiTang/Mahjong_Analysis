from typing import List, Optional, Tuple

import re

from fileProcess import RoundState, States
from mjanalyzer_local import annotate_states_shanten

STATES = States()


def _parse_step_line(line: str) -> List[str]:
    # e.g. "* 24. S M 211 [S P]"
    return line.replace(".", "").replace("* ", "").strip().split(' ')

def processAction(ref: RoundState, step_data: List[str]) -> Optional[str]:
    ref.stepData = step_data
    action_str = step_data[2]
    try:
        actor = STATES.get_player(ref, step_data[1])#主角

        if action_str == 'M':  # 摸牌，進手排
            actor.tiles.append(step_data[3])
            ref.mountainCount-=1

        if action_str == 'HD':  # 打牌，捨手排 去牌池
            actor.tiles.remove(step_data[3])
            ref.abandonTiles.append(step_data[3])

        if action_str == 'MD':  # 摸切 不加入手排 去牌池
            actor.tiles.remove(step_data[3])
            ref.abandonTiles.append(step_data[3])

        if action_str == 'P':  # 碰
            ref.abandonTiles.remove(step_data[3])
            actor.tiles.append(step_data[3])

        if action_str in ('E', 'EM', 'EL', 'ER'): # 吃
            ref.abandonTiles.remove(step_data[4])
            actor.tiles.append(step_data[4])

        if action_str in ('G', 'UG', 'HG', 'AG'):  # 明槓、暗槓、加槓
            if len(step_data) > 3 and step_data[3] in actor.tiles:
                actor.tiles.remove(step_data[3])

        if action_str == 'H' or action_str == 'SM':  # Winner or 自摸
            return step_data[1]
    except Exception:
        print("Err")
        pass

    return None


def processFile(filename: str) -> None:
    STATES.clear_all()

    step_rows: List[List[str]] = []
    river_tiles: List[str] = []

    with open(filename, "r", encoding='utf-16-le') as f:
        for line in f:
            if re.match(r'\*\s*\d+\.', line):
                step_rows.append(_parse_step_line(line))
            if "SQRWALL" in line:
                river_tiles = line.replace("* SQRWALL ", "").strip().split(' ')

    if not step_rows or not river_tiles:
        return

    STATES.playerBank = step_rows[0][1]
    STATES.winnerLoc = STATES.playerBank

    initial = RoundState(stepId=0, stepStr="START")
    for i in range(0, min(65, len(river_tiles))):
        if i < 17:
            target = 0
        elif i < 33:
            target = 1
        elif i < 49:
            target = 2
        else:
            target = 3
        initial.player[target].tiles.append(river_tiles[i])

    STATES.appendRoundWithData(initial)

    for idx, step_data in enumerate(step_rows, start=1):
        prev_state = STATES.state[-1]
        step_str = ' '.join(step_data[1:])
        next_state = prev_state.clone(step_id=idx, step_str=step_str)
        winner = processAction(next_state, step_data)
        if winner:
            STATES.winnerLoc = winner
        STATES.appendRoundWithData(next_state)
    annotate_states_shanten(STATES)


def parse_list(cards: List[str]) -> str:
    type_dict = {0: '', 1: 'm', 2: 'p', 3: 's', 4: 'z'}
    out = ""
    for card in cards:
        card_num = int(card)
        out += f"{int(card_num / 10 % 10)}{type_dict[int(card_num / 100)]}"
    return out


def strCard(card: int) -> str:
    type_dict = {0: '花', 1: '萬', 2: '筒', 3: '條', 4: '字'}
    return f"{int(card / 10 % 10)}{type_dict[int(card / 100)]}"


def format_step_action(step_data: List[str], loc_map, action_map) -> str:
    text = ' '.join(step_data)
    try:
        actor = loc_map[STATES.player_index_from_loc(step_data[0])]
        action_text = action_map.get(step_data[1], step_data[1])
        tile_text = strCard(int(step_data[2])) if len(step_data) > 2 and step_data[2].isdigit() else ''
        return f"{text} ({actor} {action_text} {tile_text})"
    except Exception:
        print("Err")
        return text

#for api
def getRound(filename: str)-> States:
    processFile(filename)
    if not STATES.state:
        raise ValueError("filename?") 
    return STATES

