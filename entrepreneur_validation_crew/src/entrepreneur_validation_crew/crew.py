from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API credentials
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = "reddit_agent"

# Import Reddit Search tools from langchain_community
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.tools.reddit_search.tool import RedditSearchSchema
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper

# For Twitter scraping (using Selenium & BeautifulSoup)
from selenium import webdriver
from bs4 import BeautifulSoup
import time


@CrewBase
class EntrepreneurValidationCrew:
    """EntrepreneurValidationCrew crew"""

    @agent
    def controller_agent(self) -> Agent:
        return Agent(config=self.agents_config["controller_agent"])

    @agent
    def reddit_agent(self) -> Agent:
        reddit_search = RedditSearchRun(
            api_wrapper=RedditSearchAPIWrapper(
                reddit_client_id=client_id,
                reddit_client_secret=client_secret,
                reddit_user_agent=user_agent,
            )
        )
        return Agent(
            config=self.agents_config["reddit_agent"],
            tools=[reddit_search],
            verbose=True,
        )

    @agent
    def twitter_agent(self) -> Agent:
        return Agent(config=self.agents_config["twitter_agent"])

    @agent
    def report_agent(self) -> Agent:
        return Agent(config=self.agents_config["report_agent"])

    @task
    def gather_input_task(self) -> Task:
        return Task(config=self.tasks_config["gather_input_task"])

    @task
    def reddit_search_task(self) -> Task:
        return Task(config=self.tasks_config["reddit_search_task"])

    @task
    def twitter_scraping_task(self) -> Task:
        return Task(config=self.tasks_config["twitter_scraping_task"])

    @task
    def generate_report_task(self) -> Task:
        return Task(config=self.tasks_config["generate_report_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    # Twitter scraping function
    def scrape_twitter(self, query: str):
        driver = webdriver.Chrome(
            executable_path="/opt/homebrew/Caskroom/chromedriver/129.0.6668.58/chromedriver-mac-arm64/chromedriver"
        )
        driver.get(f"https://twitter.com/search?q={query}&src=typed_query")
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        tweets = soup.find_all("article", {"role": "article"})

        tweet_texts = []
        for tweet in tweets:
            tweet_content = tweet.find("div", {"lang": True})
            if tweet_content:
                tweet_texts.append(tweet_content.text)

        driver.quit()
        return tweet_texts
