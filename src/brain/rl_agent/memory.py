from collections import deque 
from dataclasses import dataclass, field
import logging 
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory")

@dataclass
class MemoryBatch:
    states: list = field(default_factory=list)               # States  (s):                The encoder embeddings (768-dim vectors)
    actions: list = field(default_factory=list)              # Actions (a):                The IDs of functions/controllers picked by the Actor Network
    log_probabilities: list = field(default_factory=list)    # log-probabilities (log pi): The log probabilities of the action picked by the Actor Network
    values: list = field(default_factory=list)               # Values (V):                 The Critic Network prediction of the reward for that state
    rewards: list = field(default_factory=list)              # Rewards (r):                The actual feedback (+1, -1, etc)

class Memory:
    def __init__(self, memory_size: int = 100, batch_size: int = 5):
        self.memory_size = memory_size
        self.batch_size = batch_size   # Batch size should be at least bigger than 100 as this batch will then be split into mini batches
        
        self.full = False
        self.processed_batches = []
        self.memory_batch = list[MemoryBatch]          
    
    def get_buffer_status(self):
        return self.full
    
    def generate_minibatches(self):
        pass
        
    
    def reset_batch(self):
        logger.info("Memory buffer was reset")

        self.states = []
        self.actions = []
        self.log_probabilities = []
        self.values = []
        self.rewards = []   

        self.full = False 
    
    def append_memory(self, state, action, log, value, reward):
        logger.info("Appending a memory unit to the memory buffer")
        self.states.append(state)
        self.actions.append(action)
        self.log_probabilities.append(log)
        self.values.append(value)
        self.rewards.append(reward)
        
        if len(self.states) == self.memory_size:
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
    


            
            
