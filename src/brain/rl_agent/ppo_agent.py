from brain.rl_agent.actor_model import ActorNetwork 
from brain.rl_agent.critic_model import CriticNetwork
from memory.batch_memory import BatchMemory
from schemas.memory_schemas import InferenceSchema

from torch import nn
import torch 
import logging 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppo_agent")


class PPOAgent:
    def __init__(self, in_embed_dim: int, out_embed_di: int):
        self.actor_network = ActorNetwork(in_embed_dim=in_embed_dim, out_embed_dim=out_embed_di)
        self.critic_network = CriticNetwork(in_embed_dim=in_embed_dim)
        self.memory = BatchMemory(batch_size=10)
    
    def store_memory(self, inference_record: InferenceSchema):
        logger.info("Storing Inference memory for the PPOAgent")
        self.memory.store_inference(inference_record=inference_record)

    def select_action(self, state_embedding):
        with torch.no_grad():

            dist = self.actor_network(state_embedding)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = self.critic_network(state_embedding)

        return action.item(), log_prob, value