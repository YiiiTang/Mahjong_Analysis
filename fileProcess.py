from dataclasses import dataclass, field
from typing import List

#player's state(tiles...) in each round
@dataclass
class playerState:
    tiles: List[str] = field(default_factory=list)
    shantenCount: int = -99 #if uncalculated

    def clone(self) -> "playerState":
        return playerState(tiles=self.tiles.copy(), shantenCount=-99)

#each Round's state
@dataclass
class RoundState:
    stepStr: str = ""
    stepData: List[str] = field(default_factory=list)#1 主角 2 動作 3..10 排
    stepId: int = 0
    mountainCount: int = 56
    player: List[playerState] = field(default_factory=lambda: [playerState() for _ in range(4)])
    abandonTiles: List[str] = field(default_factory=list)

    def clone(self, step_id: int, step_str: str) -> "RoundState":
        return RoundState(
            stepStr=step_str,
            stepId=step_id,
            mountainCount= self.mountainCount,
            player=[p.clone() for p in self.player],
            abandonTiles=self.abandonTiles.copy(),
        )


#a state of 一局
@dataclass
class States:
    state: List[RoundState] = field(default_factory=list)
    playerBank: str = ""
    winnerLoc: str = "E"

    def clear_all(self) -> None:
        self.state.clear()
        self.playerBank = ""
        self.winnerLoc = "E"

    def appendRound(self) -> int:
        self.state.append(RoundState(stepId=len(self.state)))
        return len(self.state) - 1

    def appendRoundWithData(self, state: RoundState) -> int:
        self.state.append(state)
        return len(self.state) - 1

    def steps_count(self) -> int:
        # state[0] is initial tiles before any step action
        return max(0, len(self.state) - 1)

    def player_index_from_loc(self, char: str) -> int:
        order = {'E': 0, 'S': 1, 'W': 2, 'N': 3}
        if not self.playerBank:
            return -1
        if char == self.playerBank:
            return 0
        return (order[char] - order[self.playerBank]) % 4

    def get_player(self, ref: RoundState, loc_char: str) -> playerState:
        return ref.player[self.player_index_from_loc(loc_char)]

