from pydantic import BaseModel 

class MemorySchema(BaseModel):
    command: str 
    state_embedding: list[float]
    action_id: str 
    log_prob: float 
    value_pred: float 
    reward: float = 0.0 
    timestamp: float