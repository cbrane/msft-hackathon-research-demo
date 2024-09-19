from crewai import Crew, Task, Agent
from crewai.project.crew_base import CrewBase
import yaml

class EntrepreneurValidationCrew(CrewBase):
    def __init__(self, tasks_config=None):
        super().__init__()
        self.tasks_config = tasks_config or self.load_default_config()

    def load_default_config(self):
        # Load default configuration if none is provided
        with open('config/tasks.yaml', 'r') as file:
            return yaml.safe_load(file)

    def crew(self):
        # Create agents
        agents = self.create_agents()

        # Create tasks
        tasks = self.create_tasks(agents)

        # Create the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )

        return crew

    def create_agents(self):
        # Implement agent creation logic here
        # You might want to load this from a config file as well
        pass

    def create_tasks(self, agents):
        # Create tasks based on the tasks_config
        tasks = []
        for task_config in self.tasks_config['tasks']:
            task = Task(
                description=task_config['description'],
                agent=agents[task_config['agent']],
                # Add other task parameters as needed
            )
            tasks.append(task)
        return tasks

    def kickoff(self, inputs):
        return self.crew().kickoff(inputs=inputs)
