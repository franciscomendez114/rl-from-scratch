import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )

        self.action_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, input_dim)

        features = self.backbone(x)


        return self.action_head(features), self.value_head(features)


def discounted_rewards(rewards: np.ndarray, gamma: float = 0.99):
    # rewards: (T,)

    discounted_returns = np.zeros_like(rewards)

    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        discounted_returns[t] = running


    return discounted_returns / 100 # make it easier to learn, since gradients are more stable.
        

def update_model(model, optimizer, state, actions, rewards, gamma=0.99):
    # Reward shape (T,), state shape (T, 4), actions shape (T,)
    # compute discounted
    T = len(rewards)
    returns = torch.tensor(discounted_rewards(rewards, gamma), dtype=torch.float32)  # (T,)

    action_logits, value_logits = model(state) # action_logits: (T, 2), value_logits: (T, 1)

    # want to both maximize the expected the return and minimize the difference between the predicted value and the actal return

    advantage = returns - value_logits.detach().squeeze(-1)  # (T,)

    # normalize advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    value_loss = F.mse_loss(value_logits.squeeze(-1), returns)  # (T,)

    cross_entropy = F.cross_entropy(action_logits, torch.tensor(actions), reduction='none') # (T,)

    loss = torch.mean(cross_entropy * advantage) + 0.5 * value_loss # weighing the value loss by 0.5 to balance the two objectives, since 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

model = PolicyNetwork(input_dim=4, output_dim=2)
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

env = gym.make("CartPole-v1")

def run_episode(env, model):
    obs, info = env.reset()
    done = False
    episode_length = 0

    states = []
    actions = []
    rewards = []

    while not done:
        # get action from policy network
        cur_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 4)

        with torch.no_grad():
            action_logits, _ = model(cur_state)

        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        action = action_dist.sample().item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        states.append(cur_state.squeeze(0).numpy())
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        episode_length += 1

        if terminated or truncated:
            done = True

    return np.array(states), np.array(actions), np.array(rewards), episode_length

def main():
    num_episodes = 1500
    print("Starting training...")
    for episode in range(num_episodes):
        states, actions, rewards, episode_length = run_episode(env, model)

        loss = update_model(model, optimizer, torch.tensor(states, dtype=torch.float32), actions, rewards)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Length: {episode_length}, Loss: {loss.item():.4f}")
    env.close()

    print("Running demo...")
    demo_env = gym.make("CartPole-v1", render_mode="human")
    for _ in range(3):
        run_episode(demo_env, model)
    demo_env.close()

if __name__ == "__main__":
    main()