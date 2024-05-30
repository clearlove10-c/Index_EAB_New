import numpy as np

from index_advisor_selector.index_selection.swirl_selection.stable_baselines.common.vec_env import VecEnv


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()

        # 54 + 50 * size + size + size + 1 + 1 + 1 + 1
        # 54 + 50 * 18 + 18 + 18 + 1 + 1 + 1 + 1
        # action_status, workload_embedding, costs_per_query, frequencies,
        # episode_budget, current_storage_consumption, initial_cost, current_cost

        # obs[:, 54: 54 + 50 * 18] = np.zeros((1, 50 * 18))

        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        action_mask = None
        while not done:
            action_mask = [env.get_attr("valid_actions")[0]]
            action, state = model.predict(obs, state=state, deterministic=deterministic, action_mask=action_mask)
            obs, reward, done, _info = env.step(action)

            # obs[:, 54: 54 + 50 * 18] = np.zeros((1, 50 * 18))

            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: ' \
                                               '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
