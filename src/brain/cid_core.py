from brain.sentence_encoder import CommandLanguageModel 
from brain.rl_agent.ppo_agent import PPOAgent
from memory.omni_memory import OmniMemoryModule

import torch 
import logging 
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cid_core")


class CID:
    def __init__(self, 
                 encoder_model_name: str, 
                 in_embedding_dim: int = 768,
                 out_embedding_dim: int = 3,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 backend: str = "torch",
                 model_kwargs: dict = {"dtype": "float32"}):
        
        self.long_memory = OmniMemoryModule(
            
        )

        self.encoder = CommandLanguageModel(
            device = device,
            model_name = encoder_model_name,
            backend = backend, 
            model_kwargs = model_kwargs
        )
        
        self.agent = PPOAgent(
            in_embed_dim = in_embedding_dim,
            out_embed_di = out_embedding_dim
        )
        
        self._start_up()
    
    def _start_up(self):
        self.encoder.start_up()

    
    def _read_command(self, command: str):
        encoded = self.encoder.encode_command(command=command)
        return encoded
    

    def run_command(self, command: str):
        state_embedding = self._read_command(command=command)
        pred_action = self.agent.select_action(state_embedding=state_embedding)

        return pred_action



