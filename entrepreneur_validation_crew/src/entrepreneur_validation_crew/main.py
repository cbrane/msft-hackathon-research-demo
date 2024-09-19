#!/usr/bin/env python
import sys
import os
import yaml

# Use relative import for crew
from .crew import EntrepreneurValidationCrew

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def run():
    # Load tasks configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_config = load_config(os.path.join(current_dir, 'config', 'tasks.yaml'))
    
    # Prepare inputs (you might want to adjust this based on your actual input structure)
    inputs = {
        "business_idea": "A subscription box for exotic fruits",
        "target_audience": "Health-conscious foodies",
        "unique_value_proposition": "Discover rare fruits from around the world, delivered monthly"
    }
    
    # Create and run the crew
    crew = EntrepreneurValidationCrew(tasks_config=tasks_config)
    result = crew.kickoff(inputs=inputs)
    print(result)

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        EntrepreneurValidationCrew().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        EntrepreneurValidationCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and return the results.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        EntrepreneurValidationCrew().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a command: run, train, replay, or test.")
    elif sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "replay":
        replay()
    elif sys.argv[1] == "test":
        test()
    else:
        print("Invalid command.")
