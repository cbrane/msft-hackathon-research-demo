#!/usr/bin/env python
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from entrepreneur_validation_crew.crew import EntrepreneurValidationCrew


# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Run the crew.
    """
    inputs = {"topic": "AI LLMs"}
    EntrepreneurValidationCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        EntrepreneurValidationCrew().crew().train(
            n_iterations=int(sys.argv[2]), filename=sys.argv[3], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        EntrepreneurValidationCrew().crew().replay(task_id=sys.argv[2])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and return the results using GPT-4o-mini model.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        import os
        from dotenv import load_dotenv

        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        EntrepreneurValidationCrew().crew().test(
            n_iterations=int(sys.argv[2]),
            openai_model_name="gpt-4o-mini",  # Updated to use GPT-4o-mini model
            inputs=inputs,
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a command: run, train, replay, or test.")
    elif sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "train":
        if len(sys.argv) < 4:
            print("Usage: python main.py train <n_iterations> <filename>")
        else:
            train()
    elif sys.argv[1] == "replay":
        if len(sys.argv) < 3:
            print("Usage: python main.py replay <task_id>")
        else:
            replay()
    elif sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("Usage: python main.py test <n_iterations>")
        else:
            test()
    else:
        print("Invalid command.")
