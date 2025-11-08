from train_env import CaseClosedEnv
import random

if __name__ == "__main__":
    env = CaseClosedEnv(as_player=1)
    for ep in range(3):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = random.randint(0, 7)
            obs, reward, done, info = env.step(action)
            total += reward
        print(f"Episode {ep}: total_reward={total:.3f}, result={info['result']}")
