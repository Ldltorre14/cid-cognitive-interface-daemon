from agent.agent_wrapper import AgentWrapper
from brain.cid_core import CID


if __name__=="__main__":
    agent = CID()

    while True:
        user_input = input("Input your command: ")
        result = agent.run_command(command=user_input)

        print(result)