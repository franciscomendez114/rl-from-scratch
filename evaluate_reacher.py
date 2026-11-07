import gymnasium as gym
import numpy as np
import torch

from reacher_ppo import PolicyNetwork, run_episode

NUM_EPISODES = 20
SEED = 0


def main():
    # Placeholder buffers so load_state_dict can fill in the saved normalization stats
    model = PolicyNetwork(input_dim=11, output_dim=4, state_means=torch.zeros(11), state_stds=torch.ones(11))
    model.load_state_dict(torch.load("reacher_ppo.pt"))
    model.eval()

    env = gym.make("Reacher-v5")
    env.reset(seed=SEED)  # seeds the env RNG; later resets inside run_episode continue from it

    episode_rewards = []
    for episode in range(NUM_EPISODES):
        _, _, _, rewards, _, _, _, _, _ = run_episode(env, model, inference=True)
        episode_rewards.append(rewards.sum())
        print(f"Episode {episode + 1}: {rewards.sum():.2f}")

    env.close()

    print(f"Greedy reward over {NUM_EPISODES} episodes: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Reacher-v5 reward_threshold: {gym.spec('Reacher-v5').reward_threshold}")


if __name__ == "__main__":
    main()
