from collections import deque 
import logging 
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory")


class Memory:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.full = False
        self.processed_batches = []

        self.states = []               # States  (s):                The encoder embeddings (768-dim vectors)
        self.actions = []              # Actions (a):                The IDs of functions/controllers picked by the Actor Network
        self.log_probabilities = []    # log-probabilities (log pi): The log probabilities of the action picked by the Actor Network
        self.values = []               # Values (V):                 The Critic Network prediction of the reward for that state
        self.rewards = []              # Rewards (r):                The actual feedback (+1, -1, etc)
    
    def get_buffer_status(self):
        return self.full
    
    def reset_batch(self):
        logger.info("Memory buffer was reset")

        self.states = []
        self.actions = []
        self.log_probabilities = []
        self.values = []
        self.rewards = []   

        self.full = False 
    
    def append_memory(self, state, action, log, value, reward):
        logger.info("Appending a memory to the memory buffer")
        self.states.append(state)
        self.actions.append(action)
        self.log_probabilities.append(log)
        self.values.append(value)
        self.rewards.append(reward)
        
        if len(self.states) == self.batch_size:
            self.full = True
    
    def get_generalized_advantage_estimation(self, next_value, gamma_value, lambda_value):
        advantages = []
        target_returns = []
        running_advantage = 0
        rewards, values = torch.tensor(self.rewards), torch.tensor(self.values)

        for t in range(len(self.rewards) - 1, -1, -1):
            td_error = rewards[t] + (gamma_value * next_value) - values[t]
            advantage_t = td_error + (gamma_value * lambda_value * running_advantage) 

            running_advantage = advantage_t
            next_value = values[t]

            advantages.append(advantage_t)
            target_returns.append(advantage_t + values[t])
        
        return target_returns[::-1], advantages[::-1]        
    


            
            
