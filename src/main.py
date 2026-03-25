from agent.agent_wrapper import AgentWrapper



if __name__=="__main__":
    agent = AgentWrapper()

    while True:
        user_input = input("Input your command: ")
        result = agent.run(user_input=user_input)

        print(result)