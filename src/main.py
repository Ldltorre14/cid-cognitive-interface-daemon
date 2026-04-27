from agent.agent_wrapper import AgentWrapper
from brain.cid_core import CID
from config import model_name, device


if __name__=="__main__":
    agent = CID(
        encoder_model_name=model_name,
        in_embedding_dim=768, 
        out_embedding_dim=3,
        device=device,
        backend="torch",
        model_kwargs={"dtype":"float32"}
    )

    while True:
        user_input = input("Input your command: ")
        result = agent.run_command(command=user_input)

        print(result)