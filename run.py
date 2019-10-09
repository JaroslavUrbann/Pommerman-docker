from agent import Agent
from docker_agent_runner import DockerAgentRunner


class DockerAgent(DockerAgentRunner):

    def __init__(self):
        self._agent = Agent()

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    agent = DockerAgent()
    agent.run()


if __name__ == "__main__":
    main()
