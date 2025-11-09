# Case Closed Agent Template

### Explanation of Files

This template provides a few key files to get you started. Here's what each one does:

#### `agent.py`
**This is the most important file. This is your starter code, where you will write your agent's logic.**

*   DO NOT RENAME THIS FILE! Our pipeline will only recognize your agent as `agent.py`.
*   It contains a fully functional, Flask-based web server that is already compatible with the Judge Engine's API.
*   It has all the required endpoints (`/`, `/send-state`, `/send-move`, `/end`). You do not need to change the structure of these.
*   Look for the `send_move` function. Inside, you will find a section marked with comments: `# --- YOUR CODE GOES HERE ---`. This is where you should add your code to decide which move to make based on the current game state.
*   Your agent can return moves in the format `"DIRECTION"` (e.g., `"UP"`, `"DOWN"`, `"LEFT"`, `"RIGHT"`) or `"DIRECTION:BOOST"` (e.g., `"UP:BOOST"`) to use a speed boost.

#### `requirements.txt`
**This file lists your agent's Python dependencies.**

*   Don't rename this file either.
*   It comes pre-populated with `Flask` and `requests`.
*   If your agent's logic requires other libraries (like `numpy`, `scipy`, or any other package from PyPI), you **must** add them to this file.
*   When you submit, our build pipeline will run `pip install -r requirements.txt` to install these libraries for your agent.

#### `judge_engine.py`
**A copy of the runner of matches.**

*   The judge engine is the heart of a match in Case Closed. It can be used to simulate a match.
*   The judge engine can be run only when two agents are running on ports `5008` and `5009`.
*   We provide a sample agent that can be used to train your agent and evaluate its performance.

#### `case_closed_game.py`
**A copy of the official game state logic.**

*   Don't rename this file either.
*   This file contains the complete state of the match played, including the `Game`, `GameBoard`, and `Agent` classes.
*   While your agent will receive the game state as a JSON object, you can read this file to understand the exact mechanics of the game: how collisions are detected, how trails work, how boosts function, and what ends a match. This is the "source of truth" for the game rules.
*   Key mechanics:
    - Agents leave permanent trails behind them
    - Hitting any trail (including your own) causes death
    - Head-on collisions: the agent with the longer trail survives
    - Each agent has 3 speed boosts (moves twice instead of once)
    - The board has torus (wraparound) topology
    - Game ends after 500 turns or when one/both agents die

#### `sample_agent.py`
**A simple agent that you can play against.**

*   The sample agent is provided to help you evaluate your own agent's performance. 
*   In conjunction with `judge_engine.py`, you should be able to simulate a match against this agent.

#### `local-tester.py`
**A local tester to verify your agent's API compliance.**

*   This script tests whether your agent correctly implements all required endpoints.
*   Run this to ensure your agent can communicate with the judge engine before submitting.

#### `Dockerfile`
**A copy of the Dockerfile your agent will be containerized with.**

*   This is a copy of a Dockerfile. This same Dockerfile will be used to containerize your agent so we can run it on our evaluation platform.
*   It is **HIGHLY** recommended that you try Dockerizing your agent once you're done. We can't run your agent if it can't be containerized.
*   There are a lot of resources at your disposal to help you with this. We recommend you recruit a teammate that doesn't run Windows for this. 

#### `.dockerignore`
**A .dockerignore file doesn't include its contents into the Docker image**

*   This `.dockerignore` file will be useful for ensuring unwanted files do not get bundled in your Docker image.
*   You have a 5GB image size restriction, so you are given this file to help reduce image size and avoid unnecessary files in the image.

#### `.gitignore`
*   A standard configuration file that tells Git which files and folders (like the `venv` virtual environment directory) to ignore. You shouldn't need to change this.


### Testing your agent:
**Both `agent.py` and `sample_agent.py` come ready to run out of the box!**

*   To test your agent, you will likely need to create a `venv`. Look up how to do this. 
*   Next, you'll need to `pip install` any required libraries. `Flask` is one of these.
*   Finally, in separate terminals, run both `agent.py` and `sample_agent.py`, and only then can you run `judge_engine.py`.
*   You can also run `local-tester.py` to verify your agent's API compliance before testing against another agent.


### Disclaimers:
* There is a 5GB limit on Docker image size, to keep competition fair and timely.
* Due to platform and build-time constraints, participants are limited to **CPU-only PyTorch**; GPU-enabled versions, including CUDA builds, are disallowed. Any other heavy-duty GPU or large ML frameworks (like Tensorflow, JAX) will not be allowed.
* Ensure your agent's `requirements.txt` is complete before pushing changes.
* If you run into any issues, take a look at your own agent first before asking for help.

### Continuing training from a saved checkpoint
1. Archive the current model (optional but recommended): `cp model.pth logs/checkpoints/my_snapshot.pth`.
2. Create a JSON override for opponent weights (e.g., `configs/cutter35.json` with 35% cutter weight).
3. Resume training: `python -m dql.train --config configs/cutter35.json --episodes 500 --resume logs/checkpoints/my_snapshot.pth --checkpoint model_cutter35.pth --tag cutter35-run1`.
   * `--resume` loads the snapshot before training starts.
   * `--checkpoint` sets the output file for the updated weights.
   * `--tag` (optional) archives the new checkpoint in `logs/checkpoints/` with a manifest entry for easy reference.

### Randomized training starts
- The training environment now randomizes both agents' spawn locations/directions by default (`training.randomize_starts=True`).
- Disable this (to match the judge’s fixed opener) by adding `{"training": {"randomize_starts": false}}` to any config file.
- Randomized openings help the DQN discover general tactics; inference still uses the engine’s standard spawn points, so compatibility with the judge is unaffected.

### Scripted opponents available for training/eval
`dql/opponents.py` now exposes nine opponents you can mix via any config:
- `wall_hugger`, `cutter`, `random_inertia`, `straight_biased` (original set)
- `spiral_runner` – prioritizes smooth spiral turns to close space
- `aggressive_chaser` – always steers toward our head’s torus position
- `random_boost` – inertia-heavy driver that relentlessly pushes forward like a boost spammer
- `safe_pocket` – repeatedly moves toward the largest reachable pocket
- `mirror_player` – mimics our last heading whenever possible

Update the `opponents.names/weights` arrays in a config (see `configs/default.json` or `configs/cutter35.json`) to emphasize whichever behaviors you want the DQN to learn against.

### Training enhancements baked into the pipeline
- **Boosting-aware actions:** The DQN now learns over 8 actions (4 directions × boost/no-boost) by default. Existing checkpoints that lacked boost support still load (metadata encodes the action count).
- **Adaptive shaping:** Survival bonuses decay after turn 120 and a small reachable-area bonus (`rewards.area_bonus_scale`) rewards carving new pockets. Tune these knobs in `dql/config.py`.
- **Prioritized replay:** The buffer tracks TD-error priorities (`dqn.prioritized_alpha`, `prioritized_beta_start`, etc.) and waits for 20k warmup transitions before updates. You rarely need to change code—just adjust the config if you want a different schedule.
- **Target ensemble:** Two Polyak-averaged target networks are averaged to stabilize Q-targets under randomized starts.

### Curriculum-ready configs
Use the staged configs to guide training through focused matchups:
1. `configs/curriculum_stage1.json` – focuses on `spiral_runner`/`aggressive_chaser` (1k episodes)
2. `configs/curriculum_stage2.json` – cutter-heavy boot camp (another 1k episodes)
3. `configs/curriculum_stage3.json` – balanced nine-opponent finale (3k episodes)

Example flow:
```
python -m dql.train --config configs/curriculum_stage1.json --episodes 1000 --checkpoint model_stage1.pth --tag stage1
python -m dql.train --config configs/curriculum_stage2.json --episodes 1000 --resume logs/checkpoints/stage1.pth --checkpoint model_stage2.pth --tag stage2
python -m dql.train --config configs/curriculum_stage3.json --episodes 3000 --resume logs/checkpoints/stage2.pth --checkpoint model_stage3.pth --tag stage3
```
Evaluate between stages to pick the best checkpoint for deployment.

### Handy commands (training, eval, head-to-head)

| Purpose | Command |
| --- | --- |
| Install deps | `python3 -m pip install -r requirements.txt` |
| Start your agent (default port) | `PORT=5008 python agent.py` |
| Start friend’s agent (separate checkout) | `PORT=5009 python agent.py` |
| Judge a single match (stock) | `python judge_engine.py` |
| Judge 10 randomized matches + save boards | `python judge_engine_custom.py` (outputs into `logs/match_visuals/`) |
| Train from scratch (balanced opponents) | `python -m dql.train --config configs/default.json --episodes 2000 --checkpoint model_balanced.pth --tag balanced-run1` |
| Resume training from checkpoint | `python -m dql.train --config configs/cutter35.json --episodes 500 --resume logs/checkpoints/<snapshot>.pth --checkpoint model_cutter35.pth --tag cutter35-run2` |
| Evaluate a checkpoint (default config) | `python -m dql.eval --checkpoint model.pth` |
| Evaluate with custom config | `python -m dql.eval --config configs/cutter35.json --checkpoint model_cutter35.pth --matches 500` |
| Archive current model manually | `cp model.pth logs/checkpoints/<tag>.pth` |
| Create a new worktree for main branch (head-to-head) | `git worktree add ../main-agent origin/main` |

Feel free to script additional variants (e.g., `watch -n 1 python -m dql.eval ...`) depending on your workflow.
