import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from Atari_space_invaders import DQNetwork

gym.register_envs(ale_py)

agent = DQNetwork(input_shape=(4, 84, 84), num_actions=6)
state_dict = torch.load("data/space_invaders_dqn.bin")
agent.load_state_dict(state_dict)
agent.eval()

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env)
env = FrameStackObservation(env, 4)

state, _ = env.reset()
done = False
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    state_tensor = state_tensor.permute(0, 1, 2)
    state_tensor = state_tensor.unsqueeze(0)

    with torch.no_grad():
        action_probs = agent(state_tensor)
    action = torch.argmax(action_probs[0]).item()

    state, reward, done, _, _ = env.step(action)
