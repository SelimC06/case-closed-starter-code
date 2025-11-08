import random
from typing import Callable, Tuple, Dict, Any

from case_closed_game import Game, Direction, GameResult

class CaseClosedEnv:
    def __init__(
            self,
            as_player: int = 1,
            opponent_policy: Callable[[Game, int], Tuple[Direction, bool]] | None = None,
            survival_reward: float = 0.01
    ):
        assert as_player in (1,2)
        self.as_player = as_player
        self.opp_player = 2 if as_player == 1 else 1
        self.game = Game()
        self.survival_reward = survival_reward

        self.opponent_policy = opponent_policy if opponent_policy is not None else self._default_opponent_policy

    def reset(self) -> Dict[str, Any]:
        self.game.reset()
        return self._build_obs(self.as_player)

    def step(self, action: int):
        my_dir, my_boost = self._decode_action(action)
        opp_dir, opp_boost = self.opponent_policy(self.game, self.opp_player)

        if self.as_player == 1:
            result = self.game.step(my_dir, opp_dir, my_boost, opp_boost)
        else:
            result = self.game.step(opp_dir, my_dir, opp_boost, my_boost)
        
        done = result is not None

        if not done:
            reward = self.survival_reward
        else:
            if result == GameResult.DRAW:
                reward = 0.0
            elif (result == GameResult.AGENT1_WIN and self.as_player == 1) or (result == GameResult.AGENT2_WIN and self.as_player == 2):
                reward = 1.0
            else:
                reward = -1.0
        obs = self._build_obs(self.as_player)
        return obs, reward, done, {"result": result}

    def _decode_action(self, a: int) -> Tuple[Direction, bool]:
        if not 0 <= a <= 7:
            raise ValueError(f"Invalid action {a}, expected 0..7")
        
        boost = a >= 4
        dir_idx = a % 4
        dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        return dirs[dir_idx], boost

    def _get_agent(self, player_num:int):
        return self.game.agent1 if player_num == 1 else self.game.agent2

    def _build_obs(self, player_num: int) -> Dict[str, Any]:
        g = self.game

        board = [row[:] for row in g.board.grid]
        return {
            "board": board,
            "agent1_trail": list(g.agent1.trail),
            "agent2_trail": list(g.agent2.trail),
            "agent1_length": g.agent1.length,
            "agent2_length": g.agent2.length,
            "agent1_alive": g.agent1.alive,
            "agent2_alive": g.agent2.alive,
            "agent1_boosts": g.agent1.boosts_remaining,
            "agent2_boosts": g.agent2.boosts_remaining,
            "turn_count": g.turns,
            "player_number": player_num,
        }

    def _default_opponent_policy(self, game: Game, player_num: int) -> Tuple[Direction, bool]:

        agent = self._get_agent(player_num)
        board = game.board

        dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        random.shuffle(dirs)

        cur_dx, cur_dy = agent.direction.value

        for d in dirs:
            dx, dy = d.value
            if (dx, dy) == (-cur_dx, -cur_dy):
                continue

            head_x, head_y = agent.trail[-1]
            nx = (head_x + dx) % board.width
            ny = (head_y + dy) % board.height

            if board.get_cell_state((nx, ny)) == 0:
                return d, False

        return agent.direction, False