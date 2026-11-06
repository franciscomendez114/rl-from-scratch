import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        

def update_model(model, optimizer, init_log_probs, state, actions, returns):
    #we recieve the state: (B*T, 8), actions: (B*T,), returns: (B*T,) because we are training on a batch of episodes, where B is the batch size and T is the episode length.

    action_logits, value_logits = model(state) # action_logits: (T, 4), value_logits: (T, 1)

    # want to both maximize the expected the return and minimize the difference between the predicted value and the actal return

    advantage = returns - value_logits.detach().squeeze(-1)  # (T,)

    # normalize advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    value_loss = F.mse_loss(value_logits.squeeze(-1), returns)  # (T,)

    # find ratio of the probabiliity of the action taken by initial policy and new policy

    cur_log_probs = F.log_softmax(action_logits, dim=-1) # (T, 4)
    init_log_probs = init_log_probs.detach() # (T, 4)

    # so log(cur_prob) - log(init_prob) = log(cur_prob / init_prob) => cur_prob / init_prob = exp(log(cur_prob) - log(init_prob))
    ratios = torch.exp(cur_log_probs[range(len(actions)), actions] - init_log_probs[range(len(actions)), actions]) # (T,)

    # next clamp the ratios to be between 1 - epsilon and 1 + epsilon, where epsilon is a hyperparameter that controls how much we want to clip the ratios. This is to prevent the policy from changing too much in one update, which can lead to instability.
    
    epsilon = 0.2

    unclipped = ratios * advantage

    clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantage # (T,)

    loss = -torch.mean(torch.min(unclipped, clipped)) + 0.5 * value_loss # weighing the value loss by 0.5 to balance the two objectives, since 
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def run_episode(env, model, inference=False):
    obs, info = env.reset()

    done = False
    episode_length = 0

    states = []
    log_probs = []
    actions = []
    rewards = []


    while not done:
        # get action from policy network
        cur_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 8)

        with torch.no_grad():
            action_logits, _ = model(cur_state)

        action_probs = F.softmax(action_logits, dim=-1) # (1, 4)
        action_dist = torch.distributions.Categorical(action_probs)


        if inference:
            if cur_state[0][-1] == 1 and cur_state[0][-2] == 1:
                action = 0
            else:
                action = torch.argmax(action_probs).item() # (0, 1, 2, 3)
        else:
            action = action_dist.sample().item() # int (0, 1, 2, 3)

        next_obs, reward, terminated, truncated, info = env.step(action)
    
        states.append(cur_state.squeeze(0).numpy())
        actions.append(action)
        log_probs.append(F.log_softmax(action_logits, dim=-1).squeeze(0).numpy())
        rewards.append(reward)

        obs = next_obs
        episode_length += 1

        if terminated or truncated:
            done = True

    discounted_returns = discounted_rewards(np.array(rewards), gamma=0.99)

    return np.array(states), np.array(actions), np.array(log_probs), np.array(rewards), discounted_returns, episode_length


def train_model(model, optimizer, env, num_episodes, batch_size=4, num_batch_epochs=5):
    print("Starting training...")

    batch_states = []
    batch_actions = []
    batch_returns = []
    batch_log_probs = []
    episode_rewards = []

    reward_history = []

    for episode in range(num_episodes):
        states, actions, log_probs, rewards, discounted_returns, episode_length = run_episode(env, model)
        # states: (T, 8), actions: (T,), discounted_returns: (T,)
        batch_states.append(states)
        batch_actions.append(actions)
        batch_log_probs.append(log_probs)
        batch_returns.append(discounted_returns)
        
        episode_rewards.append(rewards.sum().item())
        reward_history.append(rewards.sum().item())

        if len(batch_states) >= batch_size:
            for _ in range(num_batch_epochs):

                update_model(model, optimizer, torch.tensor(np.vstack(batch_log_probs), dtype=torch.float32), torch.tensor(np.vstack(batch_states), dtype=torch.float32),
                            torch.tensor(np.hstack(batch_actions), dtype=torch.long), torch.tensor(np.hstack(batch_returns), dtype=torch.float32))
            
            batch_states = []
            batch_actions = []
            batch_returns = []
            batch_log_probs = []
        

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Length: {episode_length}, Reward: {np.mean(episode_rewards[-50:])}")
    
    env.close()

    return reward_history

def plot_reward_history(reward_history):

    # caluclate moving average of reward history
    window_size = 50
    moving_avg = []

    for i in range(window_size-1, len(reward_history)):
        moving_avg.append(np.mean(reward_history[i-window_size+1:i+1]))

    plt.plot([i for i in range(1, len(reward_history) + 1)], reward_history)
    plt.plot([i for i in range(window_size, len(reward_history) + 1)], moving_avg, color='r', linestyle='--', label='Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward History")

    plt.axhline(y=200, alpha=0.3, color='r', linestyle='--', label='Solved Threshold (200)')

    plt.legend()

    plt.savefig('lunar_lander_reward.png', dpi=150)

    plt.show()

def main():

    num_episodes = 2000
    batch_size = 8
    num_batch_epochs = 50
    
    model = PolicyNetwork(input_dim=8, output_dim=4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    env = gym.make("LunarLander-v3")

    reward_history = train_model(model, optimizer, env, num_episodes, batch_size, num_batch_epochs)

    plot_reward_history(reward_history)
    
    print("Running demo...")
    demo_env = gym.make("LunarLander-v3", render_mode="human")
    run_episode(demo_env, model, inference=True)
    demo_env.close()

if __name__ == "__main__":
    main()