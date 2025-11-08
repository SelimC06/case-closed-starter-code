import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Hyperparams (start with these; adjust)
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_SIZE = 100000
MIN_REPLAY = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200000  # steps
TARGET_UPDATE_FREQ = 1000  # steps
DEVICE = torch.device("cpu")  # CPU-only

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNNet(nn.Module):
    def __init__(self, in_channels=5, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # compute conv output size for 18x20 -> flatten dim approx:
        conv_out = 64 * 18 * 20
        self.fc = nn.Sequential(
            nn.Linear(conv_out + 8, 512),  # +8 for small auxiliary vector
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, img, aux):
        x = self.conv(img)
        x = torch.cat([x, aux], dim=1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

def select_action(net, state_img, state_aux, step, num_actions):
    eps = EPS_END + (EPS_START - EPS_END) * max(0, 1 - step / EPS_DECAY)
    if random.random() < eps:
        return random.randrange(num_actions)
    img_t = torch.tensor(state_img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    aux_t = torch.tensor(state_aux, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q = net(img_t, aux_t)
        return int(q.argmax(dim=1).item())

def train_step(policy_net, target_net, buffer, optimizer):
    if len(buffer) < MIN_REPLAY:
        return
    batch = buffer.sample(BATCH_SIZE)
    state_img = torch.tensor(np.array(batch.state), dtype=torch.float32).to(DEVICE)
    state_aux = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(DEVICE)  # if using aux for next
    action = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
    reward = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    next_img = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(DEVICE)
    done = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    q_values = policy_net(state_img, state_aux).gather(1, action)
    with torch.no_grad():
        q_next = target_net(next_img, state_aux).max(1)[0].unsqueeze(1)
        q_target = reward + (1 - done) * GAMMA * q_next
    loss = nn.functional.mse_loss(q_values, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Integration note: you need an env loop that provides state_img (C,H,W), state_aux, next_state, reward, done.