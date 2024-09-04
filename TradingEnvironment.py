#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

data = pd.read_csv('NVDA.csv')

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_profit = 0
        self.done = False
        return self._next_observation()
    
    def _next_observation(self):
        return self.data.iloc[self.current_step, :].values
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        
        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        self.total_profit = self.balance + (self.shares_held * current_price) - self.initial_balance
        reward = self.total_profit
        
        next_state = self._next_observation()
        return next_state, reward, self.done
    
    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Total Profit: {self.total_profit}')


# In[6]:


import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# In[7]:


import random
from collections import deque

# Hyperparameters
episodes = 500
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.1
learning_rate = 0.001
batch_size = 32
memory_size = 1000

# Initialize environment and model
env = TradingEnvironment(data)
state_size = env._next_observation().shape[0]
action_size = 3  # Hold, Buy, Sell
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Replay memory
memory = deque(maxlen=memory_size)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        target = reward
        if not done:
            target = reward + gamma * torch.max(model(next_state)).item()
        target_f = model(state)
        target_f[action] = target
        model.train()
        optimizer.zero_grad()
        loss = criterion(target_f, model(state))
        loss.backward()
        optimizer.step()

def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    state = torch.tensor(state, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        act_values = model(state)
    return torch.argmax(act_values).item()

# Train the DQN
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    while not env.done:
        action = act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            print(f"Episode {e+1}/{episodes} - Total Profit: {env.total_profit}")
            break
        
        replay()
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


# In[8]:


state = env.reset()
state = np.reshape(state, [1, state_size])

while not env.done:
    action = act(state)
    next_state, reward, done = env.step(action)
    state = np.reshape(next_state, [1, state_size])
    env.render()

print(f"Total Profit after testing: {env.total_profit}")


# In[ ]:




