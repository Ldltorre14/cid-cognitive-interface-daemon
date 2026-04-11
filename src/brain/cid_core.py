from brain.sentence_encoder import CommandLanguageModel 
from config import device, model_name
import torch
import logging 
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cid_core")


class CID:
    def __init__(self):
        self.encoder = CommandLanguageModel(
            device = device,
            model_name = model_name,
            backend = "torch", 
            model_kwargs = {"dtype": "float32"}
        )
        self.input_buffer = deque(maxlen=10)
        
        self._start_up()
    
    def _start_up(self):
        self.encoder.start_up()

    
    def _read_command(self, command: str):
        encoded = self.encoder.encode_command(command=command)
        self.input_buffer.append(encoded)
    

    def run_command(self, command: str):
        self._read_command(command=command)

        return self.input_buffer



