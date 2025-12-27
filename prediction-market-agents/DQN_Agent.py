import random as r
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

class AgentWallet:
  def __init__(self, budget, prop_rate) -> None:
    self.budget = budget
    self.balance = budget
    self.shares = {}
    self.prices = {}
    self.prop_rate = prop_rate

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 96)  # Reduce from 128 to 96
        self.fc2 = nn.Linear(96, 64)         # Reduce from 128 to 64
        self.fc3 = nn.Linear(64, action_dim)  # Output layer unchanged

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for all actions

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Priority values
        self.position = 0
        self.alpha = alpha  # Determines how much prioritization to use

    def push(self, state, action, reward, next_state, done):
        """Stores experience with max priority (new experience gets highest importance)."""
        max_priority = self.priorities.max() if self.buffer else 1.0  

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority  # Assign highest priority
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size, beta=0.4):
        """Samples batch with prioritization."""
        if len(self.buffer) == 0:
            return []

        priorities = self.priorities[:len(self.buffer)] ** self.alpha
        probs = priorities / priorities.sum()  # Normalize

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)  # Sample with priority
        experiences = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights to correct bias
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        # Unpack batch and convert to tensors
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, errors):
        """Updates the priority values based on new TD errors."""
        self.priorities[indices] = np.abs(errors) + 1e-5  # Prevents zero priority

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, name, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, 
                 batch_size=64, target_update=10, num_episodes=1, num_past_ex=50, num_splits=100):
        
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_save = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = num_episodes
        self.num_past_ex = num_past_ex
        self.num_splits = num_splits

        # Neural networks for training
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Use Prioritized Replay Buffer instead of standard replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state):
        """ Select action using epsilon-greedy policy """
        if r.random() < self.epsilon:
            return r.randint(0, self.action_dim - 1)  # Random action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()

    def train(self, cur_split, beta=0.4):
        """ Train the Q-network using Prioritized Experience Replay """
        if len(self.memory) < self.batch_size:
            return  # Skip training if not enough samples

        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, beta)

        # Move tensors to device
        states, actions, rewards, next_states, dones, weights = (
            states.to(self.device), actions.to(self.device), 
            rewards.to(self.device), next_states.to(self.device), 
            dones.to(self.device), weights.to(self.device)
        )

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD error (absolute difference between Q-values and target)
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)

        # Compute loss with importance sampling correction
        loss = (weights * (q_values - target_q_values) ** 2).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon *= self.epsilon_decay
        #self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-4.0 * cur_split / self.num_splits)

def train_dqn_on_data(agent, data, times, cur_split, budget=1000):
    """
    Trains the DQN on provided market data.

    :param data: A pandas DataFrame where each row is a market state.
    """

    agent.epsilon = agent.epsilon_save
    episodes = agent.num_episodes
    num_past_ex = agent.num_past_ex
    def_amount = budget / 50

    for episode in range(episodes):
      total_reward = 0
      wallet = AgentWallet(budget, 1)
      i = 0
      # Loop through historical market data row by row
      while i < len(times):
        cur_time = times.iloc[i]
        cur_idxs = np.where(data.index == cur_time)[0]
        if isinstance(cur_idxs, int):
          cur_idxs = [cur_idxs]
        num_idxs = len(cur_idxs)
        for idx in cur_idxs:
          if idx < num_past_ex or idx == len(data) - 1:
            continue
          cur_market = data.iloc[idx, 5]
          old_market = data.iloc[idx - num_past_ex, 5]
          next_market = data.iloc[idx + 1, 5]
          if cur_market != old_market or cur_market != next_market:
            continue
          for j in range(2):
            cur_price = data.iloc[idx, j]
            row_idxs = list(range(idx - num_past_ex, idx))
            col_idxs = [j] + list(range(2, 5))
            if not cur_market in wallet.shares:
              wallet.shares[cur_market] = [0, 0]
            cur_shares = wallet.shares[cur_market][j]
            cur_balance = wallet.balance

            raw_data = data.iloc[row_idxs, col_idxs].values
            mean_5 = np.mean(raw_data[-num_past_ex//10:], axis=0).tolist()
            mean_10 = np.mean(raw_data[-num_past_ex//5:], axis=0).tolist()
            mean_25 = np.mean(raw_data[-num_past_ex//2:], axis=0).tolist()
            mean_50 = np.mean(raw_data[:], axis=0).tolist()
            min_vals = np.min(raw_data, axis=0).tolist()
            max_vals = np.max(raw_data, axis=0).tolist()

            state = mean_5 + mean_10 + mean_25 + mean_50 + min_vals + max_vals + [cur_price, cur_balance, cur_shares]  # Convert row to numpy array

            action = agent.select_action(state)
                  
            req_amount = def_amount * abs(action - (agent.action_dim // 2)) * wallet.prop_rate
            req_shares = round(req_amount/cur_price)
            req_cost = req_shares * cur_price

            if action - (agent.action_dim // 2) >= 0 and req_cost <= cur_balance:
              wallet.balance -= req_cost
              wallet.shares[cur_market][j] += req_shares
            elif action - (agent.action_dim // 2) < 0 and req_shares <= cur_shares:
              wallet.balance += req_cost
              wallet.shares[cur_market][j] -= req_shares

            next_price = data.iloc[idx + 1, j]
            row_idxs = list(range(idx - num_past_ex + 1, idx + 1))

            raw_data = data.iloc[row_idxs, col_idxs].values
            mean_5 = np.mean(raw_data[-num_past_ex//10:], axis=0).tolist()
            mean_10 = np.mean(raw_data[-num_past_ex//5:], axis=0).tolist()
            mean_25 = np.mean(raw_data[-num_past_ex//2:], axis=0).tolist()
            mean_50 = np.mean(raw_data[:], axis=0).tolist()
            min_vals = np.min(raw_data, axis=0).tolist()
            max_vals = np.max(raw_data, axis=0).tolist()

            next_state = mean_5 + mean_10 + mean_25 + mean_50 + min_vals + max_vals + [next_price, wallet.balance, wallet.shares[cur_market][j]] # Next row as next state

            reward = wallet.balance + wallet.shares[cur_market][j] * next_price - (cur_balance + cur_shares * cur_price)

            if idx == len(data) - 2:
              done = True
            else:
              done = data.iloc[idx + 2, 5] != cur_market  # End when reaching last row
            agent.memory.push(state, action, reward, next_state, done)

            total_reward += reward

            agent.train(cur_split)

            if done:
              wallet.balance += wallet.shares[cur_market][j] * next_price
              wallet.shares[cur_market][j] = 0

        i += num_idxs
    
    agent.epsilon_save = agent.epsilon


    #print(f"Training money made: {wallet.balance - wallet.budget}, Total Reward = {total_reward}")

    return agent  # Trained agent

def test_dqn_on_data(agent, data, times, budget=1000):
  num_past_ex = agent.num_past_ex
  def_amount = budget / 50
  wallet = AgentWallet(budget, 1)
  agent.epsilon = 0
  total_reward = 0
  i = 0
  while i < len(times):
    cur_time = times.iloc[i]
    cur_idxs = np.where(data.index == cur_time)[0]
    if isinstance(cur_idxs, int):
      cur_idxs = [cur_idxs]
    num_idxs = len(cur_idxs)
    for idx in cur_idxs:
      if idx < num_past_ex or idx == len(data) - 1:
        continue
      cur_market = data.iloc[idx, 5]
      old_market = data.iloc[idx - num_past_ex, 5]
      next_market = data.iloc[idx + 1, 5]
      if cur_market != old_market or cur_market != next_market:
        continue
      for j in range(2):
        cur_price = data.iloc[idx, j]
        row_idxs = list(range(idx - num_past_ex, idx))
        col_idxs = [j] + list(range(2, 5))
        if not cur_market in wallet.shares:
          wallet.shares[cur_market] = [0, 0]
        if not cur_market in wallet.prices:
          wallet.prices[cur_market] = [0, 0]
            
        cur_shares = wallet.shares[cur_market][j]
        cur_balance = wallet.balance

        raw_data = data.iloc[row_idxs, col_idxs].values
        mean_5 = np.mean(raw_data[-num_past_ex//10:], axis=0).tolist()
        mean_10 = np.mean(raw_data[-num_past_ex//5:], axis=0).tolist()
        mean_25 = np.mean(raw_data[-num_past_ex//2:], axis=0).tolist()
        mean_50 = np.mean(raw_data[:], axis=0).tolist()
        min_vals = np.min(raw_data, axis=0).tolist()
        max_vals = np.max(raw_data, axis=0).tolist()

        state = mean_5 + mean_10 + mean_25 + mean_50 + min_vals + max_vals + [cur_price, cur_balance, cur_shares]  # Convert row to numpy array
        #print(state)

        action = agent.select_action(state)
        req_amount = def_amount * abs(action - (agent.action_dim // 2)) * wallet.prop_rate
        req_shares = round(req_amount/cur_price)
        req_cost = req_shares * cur_price

        if action - (agent.action_dim // 2) >= 0 and req_cost <= cur_balance:
          wallet.balance -= req_cost
          wallet.shares[cur_market][j] += req_shares
        elif action - (agent.action_dim // 2) < 0 and req_shares <= cur_shares:
          wallet.balance += req_cost
          wallet.shares[cur_market][j] -= req_shares

        next_price = data.iloc[idx + 1, j]
        wallet.prices[cur_market][j] = next_price

        reward = wallet.balance + wallet.shares[cur_market][j] * next_price - (cur_balance + cur_shares * cur_price)

        if idx == len(data) - 2:
          done = True
        else:
          done = data.iloc[idx + 2, 5] != cur_market  # End when reaching last row

        total_reward += reward

        if done:
          wallet.balance += wallet.shares[cur_market][j] * next_price
          wallet.shares[cur_market][j] = 0
        
    i += num_idxs
    if i >= len(times) - 1:
       for cur_market in wallet.shares:
          for k in range(2):
            wallet.balance += wallet.shares[cur_market][k] * wallet.prices[cur_market][k]
          wallet.shares[cur_market] = [0, 0]

  #print(f"Test money made: {wallet.balance - wallet.budget}, Total Reward = {total_reward}")
  return wallet.balance - wallet.budget

def roll_DQN(agent, data, times):
  money_made_list = []
  split_numbers = []
  split = agent.num_splits
  output_filename = f"dqn_output_{agent.name}.txt"
  for i in range(split - 1):
  #for i in range(1):
    print(f"Split n. {i+1}")
    #print(f"Split #{i+1}:")
    batch_size = len(data) // split
    j = i*batch_size
    print('Training...')
    train_dqn_on_data(agent, data, times.iloc[j:j + batch_size], i)
    print('Testing...')
    money_made = test_dqn_on_data(agent, data, times[j + batch_size:j + 2*batch_size])

    money_made_list.append(money_made)
    split_numbers.append(i+1)

  with open(output_filename, "w") as file:
    for split, money in zip(split_numbers, money_made_list):
        file.write(f"Split {split}: {money:.2f}\n")

  plt.figure(figsize=(10, 6))
  plt.plot(split_numbers, money_made_list, marker='o', linestyle='-')
  plt.title(f"{agent.name}")
  plt.xlabel('Split Number')
  plt.ylabel('Money Made ($)')
  plt.grid(True)
  plt.tight_layout()
  #plt.savefig(f"plots-by-category/dqn_performance_{agent.name}.png")
  plt.savefig(f"dqn_performance_{agent.name}.png")
  plt.close()
  #plt.show()
  
  return money_made_list

def create_agent(name, num_past=6, num_features=4, action_dim=11, num_epi=1, eps_decay=0.995, splits=100):
  state_dim = num_past*num_features+3
  agent = DQNAgent(name, state_dim, action_dim, epsilon_decay=eps_decay, num_episodes=num_epi, num_splits=splits)
  return agent

"""

econ_file = "CS229_full-data/combined_category_data/processed_economics_category.csv"
sports_file = "CS229_full-data/combined_category_data/processed_sports_category.csv"
culture_file = "CS229_full-data/combined_category_data/processed_culture_category.csv"
politics_file = "CS229_full-data/combined_category_data/processed_politics_category.csv""

"""

file = "CS229_full-data/combined_category_data/sampled_processed_economics_category.csv"

min_eps = 0.01
df = pd.read_csv(file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
sorted_timestamps = df['timestamp'].sort_values()
df.set_index('timestamp', inplace=True)
size = len(df)
name = 'economics'
half_decay = min_eps ** (2/size)
agent = create_agent(name, eps_decay=half_decay)
roll_DQN(agent, df, sorted_timestamps)