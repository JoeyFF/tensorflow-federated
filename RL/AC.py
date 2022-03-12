import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# Create the environment
env = gym.make("CartPole-v0")

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

num_actions = env.action_space.n  # 2
num_hidden_units = 128


class ActorCritic(tf.Module):
    def __init__(self, actions_num: int, hidden_num: int):
        super().__init__()
        self.common = layers.Dense(hidden_num, activation="relu")
        self.actor = layers.Dense(actions_num)
        self.critic = layers.Dense(1)

    def __call__(self, inputs, training=None, mask=None):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ one interaction between agent and environment

    :param action: a action output from actor network
    :return: state, reward, is_done
    """
    state, reward, is_done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(is_done, np.int32))


# wrap env_step as a tf function
def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action],
                             [tf.float32, tf.int32, tf.int32])


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """run a single episode to collect training data,
    including action probabilities, values, rewards

    :param initial_state: chu
    """

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    # the t-th step of one episode
    for t in tf.range(max_steps):
        # State Batch as input
        state = tf.expand_dims(state, 0)
        action_logits_t, value = model(state)

        # Sample current action based on action probabilities
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store action_probability
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, is_done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(is_done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probs, values, rewards


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    reward_num = rewards.shape[0]
    returns = tf.TensorArray(dtype=tf.float32, size=reward_num)
    discounted_sum = tf.constant(0, dtype=tf.float32)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(reward_num):
        reward = rewards[i]
        discounted_sum = reward+gamma*discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)


model = ActorCritic(num_actions, num_hidden_units)
