import requests
import sys
import time
import os
from pathlib import Path
from case_closed_game import Game, Direction, GameResult
import random


class RandomPlayer:
    def __init__(self, player_id=1):
        self.player_id = player_id

    def get_possible_moves(self):
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def get_best_move(self):
        return random.choice(self.get_possible_moves())


TIMEOUT = 4


class PlayerAgent:
    def __init__(self, participant, agent_name):
        self.participant = participant
        self.agent_name = agent_name
        self.latency = None


class Judge:
    def __init__(self, p1_url, p2_url):
        self.p1_url = p1_url
        self.p2_url = p2_url
        self.game = Game()
        self.p1_agent = None
        self.p2_agent = None
        self.game_str = ""

    def check_latency(self):
        try:
            start_time = time.time()
            response = requests.get(self.p1_url, timeout=TIMEOUT)
            end_time = time.time()
            if response.status_code == 200:
                data = response.json()
                self.p1_agent = PlayerAgent(data.get("participant", "Participant1"),
                                            data.get("agent_name", "Agent1"))
                self.p1_agent.latency = (end_time - start_time)
            else:
                return False
        except (requests.RequestException, requests.Timeout):
            return False

        try:
            start_time = time.time()
            response = requests.get(self.p2_url, timeout=TIMEOUT)
            end_time = time.time()
            if response.status_code == 200:
                data = response.json()
                self.p2_agent = PlayerAgent(data.get("participant", "Participant2"),
                                            data.get("agent_name", "Agent2"))
                self.p2_agent.latency = (end_time - start_time)
            else:
                return False
        except (requests.RequestException, requests.Timeout):
            return False

        return True

    def send_state(self, player_num):
        url = self.p1_url if player_num == 1 else self.p2_url
        state_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "player_number": player_num,
        }
        try:
            response = requests.post(f"{url}/send-state", json=state_data, timeout=TIMEOUT)
            return response.status_code == 200
        except (requests.RequestException, requests.Timeout):
            return False

    def get_move(self, player_num, attempt_number, random_moves_left):
        url = self.p1_url if player_num == 1 else self.p2_url
        params = {
            "player_number": player_num,
            "attempt_number": attempt_number,
            "random_moves_left": random_moves_left,
            "turn_count": self.game.turns,
        }
        try:
            start_time = time.time()
            response = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
            end_time = time.time()
            if player_num == 1:
                self.p1_agent.latency = (end_time - start_time)
            else:
                self.p2_agent.latency = (end_time - start_time)
            if response.status_code == 200:
                move = response.json()
                return move.get('move')
            return None
        except (requests.RequestException, requests.Timeout):
            return None

    def end_game(self, result):
        end_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "result": result.name if isinstance(result, GameResult) else str(result),
        }
        try:
            requests.post(f"{self.p1_url}/end", json=end_data, timeout=TIMEOUT)
            requests.post(f"{self.p2_url}/end", json=end_data, timeout=TIMEOUT)
            if isinstance(result, GameResult):
                if result == GameResult.AGENT1_WIN:
                    print(f"Winner: Agent 1 ({self.p1_agent.agent_name})")
                elif result == GameResult.AGENT2_WIN:
                    print(f"Winner: Agent 2 ({self.p2_agent.agent_name})")
                else:
                    print("Game ended in a draw")
            else:
                print(f"Game ended: {result}")
        except (requests.RequestException, requests.Timeout):
            return False

    def handle_move(self, move, player_num, is_random=False):
        if not isinstance(move, str):
            print(f"Invalid move format by Player {player_num}: move must be a string")
            return "forfeit"
        move_parts = move.upper().split(':')
        direction_str = move_parts[0]
        use_boost = len(move_parts) > 1 and move_parts[1] == 'BOOST'
        direction_map = {
            'UP': Direction.UP,
            'DOWN': Direction.DOWN,
            'LEFT': Direction.LEFT,
            'RIGHT': Direction.RIGHT,
        }
        if direction_str not in direction_map:
            print(f"Invalid direction by Player {player_num}: {direction_str}")
            return "forfeit"
        direction = direction_map[direction_str]
        agent = self.game.agent1 if player_num == 1 else self.game.agent2
        current_dir = agent.direction
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            print(f"Player {player_num} attempted invalid move (opposite direction). Using current direction instead.")
            direction = current_dir
            direction_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN',
                             Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        print(f"Player {player_num}'s move: {direction_str}{' (BOOST)' if use_boost else ''}{' (RANDOM)' if is_random else ''}")
        move_abbrev = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R'}
        boost_marker = 'B' if use_boost else ''
        random_marker = 'R' if is_random else ''
        self.game_str += f"{player_num}{move_abbrev[direction_str]}{boost_marker}{random_marker}-"
        return True, use_boost, direction


def board_to_string(board):
    chars = {0: ".", 1: "A"}
    lines = []
    for row in board.grid:
        line = " ".join(chars.get(cell, "A") for cell in row)
        lines.append(line)
    return "\n".join(lines)


def finalize_result(judge, result):
    return {
        "result": result,
        "game_log": judge.game_str,
        "board": board_to_string(judge.game.board),
        "turns": judge.game.turns,
    }


def run_match(game_number, seed, p1_url, p2_url):
    print(f"\n===== Starting Game {game_number} (seed {seed:.0f}) =====")
    random.seed(seed)
    judge = Judge(p1_url, p2_url)

    if not judge.check_latency():
        print("Failed to connect to one or both players")
        return None

    print(f"Player 1: {judge.p1_agent.agent_name} ({judge.p1_agent.participant})")
    print(f"Player 2: {judge.p2_agent.agent_name} ({judge.p2_agent.participant})")
    print(f"Initial latencies - P1: {judge.p1_agent.latency:.3f}s, P2: {judge.p2_agent.latency:.3f}s")

    print("Sending initial game state...")
    if not judge.send_state(1) or not judge.send_state(2):
        print("Failed to send initial state")
        return None

    p1_random = 5
    p2_random = 5

    while True:
        print(f"\n=== Turn {judge.game.turns + 1} ===")
        p1_move = None
        p2_move = None
        p1_boost = False
        p2_boost = False

        print("Requesting move from Player 1...")
        for attempt in range(1, 3):
            p1_move = judge.get_move(1, attempt, p1_random)
            if p1_move:
                validation = judge.handle_move(p1_move, 1, is_random=False)
                if validation == "forfeit":
                    print("Player 1 forfeited")
                    judge.end_game(GameResult.AGENT2_WIN)
                    print("Game String:", judge.game_str)
                    return finalize_result(judge, GameResult.AGENT2_WIN)
                elif validation:
                    p1_boost = validation[1]
                    p1_direction = validation[2]
                    break
            print(f"  Attempt {attempt} failed")
        if not p1_move or not validation:
            if p1_random > 0:
                print(f"Using random move for Player 1 ({p1_random} random moves left)")
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p1_direction], 1, is_random=True)
                p1_boost = False
            else:
                print("Player 1 has no random moves left. Forfeiting.")
                judge.end_game(GameResult.AGENT2_WIN)
                print("Game String:", judge.game_str)
                return finalize_result(judge, GameResult.AGENT2_WIN)

        print("Requesting move from Player 2...")
        for attempt in range(1, 3):
            p2_move = judge.get_move(2, attempt, p2_random)
            if p2_move:
                validation = judge.handle_move(p2_move, 2, is_random=False)
                if validation == "forfeit":
                    print("Player 2 forfeited")
                    judge.end_game(GameResult.AGENT1_WIN)
                    print("Game String:", judge.game_str)
                    return finalize_result(judge, GameResult.AGENT1_WIN)
                elif validation:
                    p2_boost = validation[1]
                    p2_direction = validation[2]
                    break
            print(f"  Attempt {attempt} failed")
        if not p2_move or not validation:
            if p2_random > 0:
                print(f"Using random move for Player 2 ({p2_random} random moves left)")
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p2_direction], 2, is_random=True)
                p2_boost = False
            else:
                print("Player 2 has no random moves left. Forfeiting.")
                judge.end_game(GameResult.AGENT1_WIN)
                print("Game String:", judge.game_str)
                return finalize_result(judge, GameResult.AGENT1_WIN)

        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        judge.send_state(1)
        judge.send_state(2)

        print(judge.game.board)
        print(f"Agent 1: Trail Length={judge.game.agent1.length}, Alive={judge.game.agent1.alive}, Boosts={judge.game.agent1.boosts_remaining}")
        print(f"Agent 2: Trail Length={judge.game.agent2.length}, Alive={judge.game.agent2.alive}, Boosts={judge.game.agent2.boosts_remaining}")

        if result is not None:
            judge.end_game(result)
            print("Game String:", judge.game_str)
            return finalize_result(judge, result)
        if judge.game.turns >= 500:
            print("Maximum turns reached")
            judge.end_game(GameResult.DRAW)
            print("Game String:", judge.game_str)
            return finalize_result(judge, GameResult.DRAW)


def main():
    print("Custom judge engine starting up, waiting for agents...")
    time.sleep(5)
    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")

    results = []
    visual_dir = Path("logs/match_visuals")
    visual_dir.mkdir(parents=True, exist_ok=True)
    for game_number in range(1, 11):
        seed = time.time() + random.random() * 1000.0
        outcome = run_match(game_number, seed, PLAYER1_URL, PLAYER2_URL)
        if outcome is None:
            results.append("FAILED")
            continue
        result_name = outcome["result"].name if isinstance(outcome["result"], GameResult) else str(outcome["result"])
        results.append(result_name)
        match_file = visual_dir / f"game{game_number}_seed{int(seed)}.txt"
        with match_file.open("w", encoding="utf-8") as fh:
            fh.write(f"Seed: {seed}\n")
            fh.write(f"Result: {result_name}\n")
            fh.write(f"Turns: {outcome['turns']}\n")
            fh.write("Game Log:\n")
            fh.write(outcome["game_log"] + "\n")
            fh.write("Final Board:\n")
            fh.write(outcome["board"] + "\n")

    print("\n===== 10-Game Summary =====")
    totals = {"AGENT1_WIN": 0, "AGENT2_WIN": 0, "DRAW": 0, "OTHER": 0}
    for idx, outcome in enumerate(results, start=1):
        print(f"Game {idx}: {outcome}")
        key = outcome if outcome in totals else "OTHER"
        totals[key] += 1
    print("Totals:")
    for key, value in totals.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
    sys.exit(0)
