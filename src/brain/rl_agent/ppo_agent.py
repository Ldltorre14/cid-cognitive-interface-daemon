from brain.rl_agent.actor_model import ActorNetwork 
from brain.rl_agent.critic_model import CriticNetwork
from brain.rl_agent.memory import Memory
import torch 
from torch import nn


class PPOAgent:
    def __init__(self):
        self.actor_network = ActorNetwork(in_embed_dim=768, out_embed_dim=3)
        self.critic_network = CriticNetwork(in_embed_dim=768)
        self.memory = Memory(batch_size=10)

    def select_action(self, state_embedding):
        with torch.no_grad():

            dist = self.actor_network(state_embedding)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = self.critic_network(state_embedding)

        return action.item(), log_prob, value