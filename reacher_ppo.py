import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio

MIN_SIGMA = 0.05

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, state_means=None, state_stds=None):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )

        self.action_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

        self.register_buffer('state_means', state_means)
        self.register_buffer('state_stds', state_stds)

    def forward(self, x):
        # x: (B, input_dim)

        if self.state_means is not None and self.state_stds is not None:
            x = (x - self.state_means) / self.state_stds.clamp(min=1e-2)

        features = self.backbone(x)

        return self.action_head(features), self.value_head(features)

def discounted_rewards(rewards: np.ndarray, gamma: float = 0.99):
    # rewards: (T,)

    discounted_returns = np.zeros_like(rewards)

    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        discounted_returns[t] = running


    return discounted_returns / 10 # make it easier to learn, since gradients are more stable.

def update_model(model, optimizer, init_log_probs, state, actions, returns):
    #we recieve the state: (B*T, 10), actions: (B*T, 2), returns: (B*T,) because we are training on a batch of episodes, where B is the batch size and T is the episode length.

    action_logits, value_logits = model(state) # action_logits: (T, 4), value_logits: (T, 1)

    # want to both maximize the expected the return and minimize the difference between the predicted value and the actal return

    advantage = returns - value_logits.detach().squeeze(-1)  # (T,)

    # normalize advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    value_loss = F.mse_loss(value_logits.squeeze(-1), returns)  # (T,)

    # find ratio of the probabiliity of the action taken by initial policy and new policy

    cur_mu = action_logits[:, ::2] # (T, 2)
    cur_sigma = F.softplus(action_logits[:, 1::2]) + MIN_SIGMA # (T, 2)
    
    # create a normal distribution with the current mu and sigma
    cur_prob_dists = Normal(loc=cur_mu, scale=cur_sigma)

    cur_log_probs = cur_prob_dists.log_prob(actions) # (T, 2)

    init_log_probs = init_log_probs.detach() # (T, 2)

    # so log(cur_prob) - log(init_prob) = log(cur_prob / init_prob) => cur_prob / init_prob = exp(log(cur_prob) - log(init_prob))
    ratios = torch.exp(cur_log_probs.sum(dim=-1) - init_log_probs.sum(dim=-1)) # (T,)

    # next clamp the ratios to be between 1 - epsilon and 1 + epsilon, where epsilon is a hyperparameter that controls how much we want to clip the ratios. This is to prevent the policy from changing too much in one update, which can lead to instability.
    
    epsilon = 0.2

    unclipped = ratios * advantage

    clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantage # (T,)

    loss = -torch.mean(torch.min(unclipped, clipped)) + 0.5 * value_loss # weighing the value loss by 0.5 to balance the two objectives, since 
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def run_episode(env, model, inference=False, record=False):
    obs, info = env.reset()

    done = False
    episode_length = 0

    states = []
    log_probs = []
    actions = []
    rewards = []
    frames = []
    stds = []
    mus = []

    steps_remaining = 50

    if record: 
        # for rendering the simulation to a gif
        frames.append(env.render())

    while not done:
        # get action from policy network
        cur_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 8)
        cur_state = torch.cat([cur_state, torch.tensor([[steps_remaining / 50]], dtype=torch.float32)], dim=-1) # (1, 9)
        
        with torch.no_grad():
            action_logits, _ = model(cur_state)

        action_gussian_means = action_logits.squeeze(0)[::2] # (4,) for the two continuous actions (mu, sigma) for each action dimension
        action_gaussian_stds = F.softplus(action_logits.squeeze(0)[1::2]) + MIN_SIGMA # (4,) for the two continuous actions (mu, sigma) for each action dimension, where it must be positive
        action_dist = Normal(loc=action_gussian_means, scale=action_gaussian_stds) # (2,) one sample for each action dimension

        if inference:
            # we just take the mean of the gaussian since this is most likely action
            unclipped_action = action_gussian_means.detach().numpy() # (2,) for the two continuous actions
        else:
            unclipped_action = action_dist.sample().detach().numpy() # (2,) one sample for each action dimension

        clipped_action = np.clip(unclipped_action, -1, 1) # (2,)

        next_obs, reward, terminated, truncated, info = env.step(clipped_action)
    
        states.append(cur_state.squeeze(0).numpy())
        actions.append(unclipped_action)
        log_probs.append(action_dist.log_prob(torch.tensor(unclipped_action)).detach().numpy())
        rewards.append(reward)
        stds.append(action_gaussian_stds.mean().item())
        mus.append(torch.abs(action_gussian_means).mean().item())
        if record:
            frames.append(env.render())
        
        obs = next_obs
        episode_length += 1
        steps_remaining -= 1

        if terminated or truncated:
            done = True

    discounted_returns = discounted_rewards(np.array(rewards), gamma=0.99)

    return np.array(states), np.array(actions), np.array(log_probs), np.array(rewards), discounted_returns, episode_length, frames, np.mean(stds), np.mean(mus)

def train_model(model, optimizer, env, num_episodes, batch_size=4, num_batch_epochs=5):
    print("Starting training...")

    batch_states = []
    batch_actions = []
    batch_returns = []
    batch_log_probs = []
    episode_rewards = []

    reward_history = []

    for episode in range(num_episodes):
        states, actions, log_probs, rewards, discounted_returns, episode_length, _, avg_sigma, avg_mu_magnitude = run_episode(env, model)
        # states: (T, 10), actions: (T, 2), discounted_returns: (T,)
        batch_states.append(states)
        batch_actions.append(actions)
        batch_log_probs.append(log_probs)
        batch_returns.append(discounted_returns)
        
        episode_rewards.append(rewards.sum().item())
        reward_history.append(rewards.sum().item())

        if len(batch_states) >= batch_size:
            for _ in range(num_batch_epochs):

                update_model(model, optimizer, torch.tensor(np.vstack(batch_log_probs), dtype=torch.float32), torch.tensor(np.vstack(batch_states), dtype=torch.float32),
                            torch.tensor(np.vstack(batch_actions), dtype=torch.float32), torch.tensor(np.hstack(batch_returns), dtype=torch.float32))
            
            batch_states = []
            batch_actions = []
            batch_returns = []
            batch_log_probs = []
        

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Length: {episode_length}, Reward: {np.mean(episode_rewards[-50:])}, Avg Sigma: {avg_sigma:.4f}, Avg Mu Magnitude: {avg_mu_magnitude:.4f}")
    
    env.close()

    return reward_history


def plot_reward_history(reward_history):

    # caluclate moving average of reward history
    window_size = 50
    moving_avg = []

    for i in range(window_size-1, len(reward_history)):
        moving_avg.append(np.mean(reward_history[i-window_size+1:i+1]))

    plt.plot([i for i in range(1, len(reward_history) + 1)], reward_history, alpha=0.3)
    plt.plot([i for i in range(window_size, len(reward_history) + 1)], moving_avg, color='r', linestyle='--', label='Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward History")

    plt.axhline(y=-3.75, alpha=0.3, color='r', linestyle='--', label='Solved Threshold')

    plt.legend()

    plt.savefig('reacher_ppo_reward.png', dpi=150)

    plt.show()

def get_global_state_stats(num_samples=1000):
    dummy_env = gym.make("Reacher-v5")
    
    states = []

    obs, _ = dummy_env.reset()

    steps_remaining = 50
    for _ in range(num_samples - 1):
        obs, _, terminated, truncated, _ = dummy_env.step(dummy_env.action_space.sample())
        # stack states
        states.append(np.array(obs.tolist() + [steps_remaining / 50]))

        steps_remaining -= 1

        if terminated or truncated:
            obs, _ = dummy_env.reset()
            steps_remaining = 50

    states = np.vstack(states)
    global_state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32) # (11,)
    global_state_std = torch.tensor(states.std(axis=0), dtype=torch.float32) # (11,)

    return global_state_mean, global_state_std

def main():

    # Notes:
    # This environment has a continuous action space.
    # input is 11 dimentional (11,), output is 4 numbers (4,) representing the four continuous actions (torque for each joint).
    SEED = 42
    torch.manual_seed(SEED)

    num_episodes = 100000
    batch_size = 10
    num_batch_epochs = 5

    state_mean, state_std = get_global_state_stats(num_samples=2000)
    
    model = PolicyNetwork(input_dim=11, output_dim=4, state_means=state_mean, state_stds=state_std)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    # Look at model performance before training
    print("Running demo before training...")
    demo_env = gym.make("Reacher-v5", render_mode="rgb_array")
    frames = []
    for _ in range(5):
        _, _, _, _, _, _, episode_frames, _, _ = run_episode(demo_env, model, inference=False, record=True)
        frames.extend(episode_frames)
    imageio.mimsave('reacher_before_training.gif', frames, duration=40, loop=0)
    demo_env.close()

    # Train the model
    env = gym.make("Reacher-v5")

    reward_history = train_model(model, optimizer, env, num_episodes, batch_size, num_batch_epochs)
    
    torch.save(model.state_dict(), "reacher_ppo.pt")

    plot_reward_history(reward_history)

    print("Running demo after training...")
    # Rinning model inference on evironment and getting rgb array to save as a gif
    demo_env = gym.make("Reacher-v5", render_mode="rgb_array")
    frames = []
    for _ in range(5):
        _, _, _, rewards, _, _, episode_frames, _, _ = run_episode(demo_env, model, inference=True, record=True)
        frames.extend(episode_frames)
        print(f"Greedy episode reward: {rewards.sum():.2f}")
    imageio.mimsave('reacher_after_training.gif', frames, duration=40, loop=0)
    demo_env.close()

if __name__ == "__main__":
    main()

