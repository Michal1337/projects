import tensorflow as tf
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt
tf.random.set_seed(1337)
np.random.seed(1337)

#Hyperparameters
num_episodes = 1000
n_units = 128
epsilon = 1.0
batch_size = 4
discount = 0.99

#https://www.gymlibrary.dev/environments/classic_control/cart_pole/
env = gym.make('CartPole-v1')
n_features = env.observation_space.shape

n_features = n_features[0]
n_actions = env.action_space.n

#https://en.wikipedia.org/wiki/Q-learning#Double_Q-learning
main_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(n_features, activation="relu"),
    tf.keras.layers.Dense(n_units, activation="relu"),
    tf.keras.layers.Dense(n_units, activation="relu"),
    tf.keras.layers.Dense(n_actions),
])
target_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(n_features, activation="relu"),
    tf.keras.layers.Dense(n_units, activation="relu"),
    tf.keras.layers.Dense(n_units, activation="relu"),
    tf.keras.layers.Dense(n_actions),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
mse = tf.keras.losses.MeanSquaredError()

class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state))
            actions.append(np.array(action))
            rewards.append(reward)
            next_states.append(np.array(next_state))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32)
    
def policy(state, epsilon):
    result = tf.random.uniform((1,))
    if result < epsilon:
        return env.action_space.sample() # Random action
    else:
        return tf.argmax(main_nn(state)[0]).numpy() # Best action
    
@tf.function
def train_step(states, actions, rewards, next_states, dones):
    next_qs = target_nn(next_states)
    max_next_qs = tf.reduce_max(next_qs, axis=-1)
    target = rewards + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, n_actions)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss 

#Training
buffer = ReplayBuffer(100000)
last_50_rewards = []
losses = []
all_rewards = []
cnt = 0
for episode in range(num_episodes+1):
    state = env.reset()[0]
    episode_reward, done = 0, False
    loss = 0
    while not done:
        state_in = tf.expand_dims(state, axis=0)
        action = policy(state_in, epsilon)
        #https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        next_state, reward, done = env.step(action)[:3] #step returns: np.array(self.state, dtype=np.float32), reward, terminated, False, {} - last False, {} not useful
        episode_reward += reward
        # Save to experience replay.
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        cnt += 1
        #Copy main_nn weights to target_nn - Double Q-learning
        if cnt % 2000 == 0:
            target_nn.set_weights(main_nn.get_weights())

        #Train neural network
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            loss += train_step(states, actions, rewards, next_states, dones)
            
    losses.append(loss)
    all_rewards.append(episode_reward)
    if (episode+1) % 50 == 0:
        print(f'Episode {episode+1}/{num_episodes} - Epsilon: {epsilon:.3f} - Reward in last 50 episodes: {np.mean(last_50_rewards):.3f}')

    if episode < 950:
        epsilon -= 0.001

    if len(last_50_rewards) == 50:
        last_50_rewards = []
    last_50_rewards.append(episode_reward)
        
env.close()

#Test the policy
def select_action(state, target_nn):
    return tf.argmax(target_nn(state)[0]).numpy()

n_episodes = 50
total_reward = 0
for episode in range(n_episodes):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    actions = []
    while not done:
        state_in = tf.expand_dims(state, axis=0)
        action = select_action(state_in, target_nn)
        actions.append(action)
        next_state, reward, done = env.step(action)[:3]
        episode_reward += reward
        state = next_state
    total_reward += episode_reward
#final reward, max is 475
total_reward/=n_episodes
print(f"{total_reward=}")