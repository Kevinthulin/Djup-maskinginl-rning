import os
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, GrayscaleObservation
import time

gym.register_envs(ale_py)

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device='cuda'):
        self.device = device
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.states = torch.zeros((capacity, 4, 84, 84), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, 4, 84, 84), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(device=self.device, dtype=torch.float32)

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        else:
            next_state = next_state.to(device=self.device, dtype=torch.float32)
        
        action = int(action)
        reward = float(reward)
        done = float(done)

        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size

class DQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x    

class SpaceInvadersDQN:
    def __init__(self,
                 env_name='ALE/SpaceInvaders-v5',
                 learning_rate=0.0001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=1000000,
                 memory_size=10000,
                 batch_size=64,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = gym.make(env_name)
        self.env = ResizeObservation(self.env, (84, 84))
        self.env = GrayscaleObservation(self.env)
        self.env = FrameStackObservation(self.env, stack_size=4)
        self.scaler = torch.amp.GradScaler('cuda')

        self.input_shape = (4, 84, 84) 
        self.action_space = self.env.action_space.n

        self.learning_start = 10000
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device

        self.memory = ReplayBuffer(capacity=100000, state_shape=(4, 84, 84))

        self.main_network = DQNetwork(self.input_shape, self.action_space).to(self.device)
        self.target_network = DQNetwork(self.input_shape, self.action_space).to(self.device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

        self._update_target_network()

    def _update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif state.device != self.device:
            state = state.to(device=self.device)

        if state.ndim == 5:
            state_tensor = state.unsqueeze(0).squeeze(0)
        elif state.ndim == 3:
            state_tensor = state.unsqueeze(0)
        else:
            state_tensor = state

        state_tensor = state_tensor.to(device=self.device)
            
        with torch.no_grad():
            q_values = self.main_network(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, frame_count):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            q_values = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]

            targets = rewards + (1 - dones) * self.gamma * next_q_values
            loss = nn.SmoothL1Loss()(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - frame_count / self.epsilon_decay)

        if frame_count % 1000 == 0:
            self._update_target_network()

    def train(self, num_episodes=1000, max_steps=10000):
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print(f"Observation space after ResizeObservation: {self.env.observation_space}")
        print(f"Observation space after FrameStackObservation: {self.env.observation_space}")
        print(f"Replay buffer state shape: {self.memory.states.shape}")

        start_time = time.time()
        self.episode_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

            if state.ndim == 3:
                state = state.unsqueeze(0)
            total_reward = 0
            frame_count = 0

            for _ in range(max_steps):
                frame_count += 1

                if len(self.memory) < self.learning_start:
                    action = self.env.action_space.sample()
                else:             
                    action = self.choose_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                if next_state.ndim == 3:
                    next_state = next_state.unsqueeze(0)
                done = terminated or truncated

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if frame_count % 4 == 0 and len(self.memory) >= self.learning_start:
                    self.replay(frame_count)
                    
                if frame_count % 10000 == 0:
                    print(f"Frame {frame_count}: Replay Buffer Size = {len(self.memory)}")
                    print(f"Frame {frame_count}: Epsilon = {self.epsilon:.4f}")

                if done:
                    break

            self.episode_rewards.append(total_reward)
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")
            
            if episode % 10 == 0:
                self._update_target_network()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / num_episodes:.2f} seconds per episode).")
        
    def test(self, num_episodes=10):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.from_numpy(state).float()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                next_state = np.array(next_state)
                done = terminated or truncated

                state = next_state
                total_reward += reward
            print("State shape:", np.array(state).shape)

            print(f"Test Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

    def save_model(self, filepath='space_invaders_dqn.bin'):
        torch.save(self.main_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='space_invaders_dqn.bin'):
        self.main_network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
    def save_rewards(self, filepath='data_analasys.npy'):
        np.save(filepath, self.episode_rewards)
        print(f"Rewards saved to {filepath}")

def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    agent = SpaceInvadersDQN(batch_size=64) 
    agent.train(num_episodes=10000)
    agent.test()
    agent.save_model()
    agent.save_rewards()

if __name__ == "__main__":
    main()