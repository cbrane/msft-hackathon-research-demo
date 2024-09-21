import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.tools.reddit_search.tool import (
    RedditSearchRun,
    RedditSearchSchema,
)
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain.tools import Tool
from selenium import webdriver
from bs4 import BeautifulSoup
import time

# Load environment variables
load_dotenv()

# API credentials
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = "reddit_agent"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EntrepreneurValidationCrew:
    def __init__(self):
        self.controller_agent = self.create_controller_agent()
        self.reddit_agent = self.create_reddit_agent()
        self.twitter_agent = self.create_twitter_agent()
        self.report_agent = self.create_report_agent()

    def create_controller_agent(self):
        return Agent(
            role="Entrepreneur Idea Consultant",
            goal="Gather business idea details and delegate search tasks to the Reddit and Twitter agents.",
            backstory="You are an expert in helping entrepreneurs refine and test their ideas against customer intent. You gather essential information about the business idea and delegate the analysis work.",
            verbose=True,
        )

    def create_reddit_agent(self):
        reddit_search = RedditSearchRun(
            api_wrapper=RedditSearchAPIWrapper(
                reddit_client_id=client_id,
                reddit_client_secret=client_secret,
                reddit_user_agent=user_agent,
            )
        )

        def search_reddit(query: str, subreddit: str = "all", limit: str = "10"):
            search_params = RedditSearchSchema(
                query=query,
                sort="relevance",
                time_filter="month",
                subreddit=subreddit,
                limit=limit,
            )
            return reddit_search.run(tool_input=search_params.dict())

        reddit_tool = Tool(
            name="Reddit Search",
            func=search_reddit,
            description="Search Reddit for relevant posts and discussions",
        )

        return Agent(
            role="Reddit Market Researcher",
            goal="Search Reddit for customer feedback, market discussions, and sentiment analysis around the target idea.",
            backstory="You specialize in finding customer intent and discussions on Reddit. You explore relevant threads and topics to gather information about the problem and the solution from a wide range of communities.",
            tools=[reddit_tool],
            verbose=True,
        )

    def create_twitter_agent(self):
        def scrape_twitter(query: str):
            driver = webdriver.Chrome()
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
            return "\n".join(tweet_texts[:10])  # Return the first 10 tweets

        twitter_tool = Tool(
            name="Twitter Scraper",
            func=scrape_twitter,
            description="Scrape Twitter for relevant tweets and discussions",
        )

        return Agent(
            role="Twitter Market Researcher",
            goal="Scrape Twitter for customer opinions, discussions, and sentiment around the target idea.",
            backstory="You are an expert in extracting relevant discussions and tweets on Twitter related to customer needs and market trends. Your insights help entrepreneurs validate their ideas based on social media sentiment.",
            tools=[twitter_tool],
            verbose=True,
        )

    def create_report_agent(self):
        return Agent(
            role="Market Analysis Report Generator",
            goal="Analyze findings from Reddit and Twitter, compile the data, and generate a detailed report on customer intent and market need.",
            backstory="You excel in turning complex market data into clear, actionable insights. Your reports help entrepreneurs make informed decisions about their product ideas.",
            verbose=True,
        )

    def run_crew(self):
        gather_input_task = Task(
            description="Gather input from the user regarding their business idea, including the problem, solution, UVP (Unique Value Proposition), demographic, and industry.",
            expected_output="A structured dataset containing the user input for the business idea, including problem, solution, demographic, and industry.",
            agent=self.controller_agent,
        )

        reddit_search_task = Task(
            description="Search Reddit for discussions relevant to the provided problem and solution.",
            expected_output="A list of Reddit posts that match the query and can be used to understand customer sentiment regarding the provided problem and solution.",
            agent=self.reddit_agent,
            context=[gather_input_task],
        )

        twitter_scraping_task = Task(
            description="Scrape Twitter for tweets discussing the user's problem, solution, and potential market.",
            expected_output="A collection of tweets related to the user's market or product, showcasing public discussions, sentiment, and customer interest.",
            agent=self.twitter_agent,
            context=[gather_input_task],
        )

        generate_report_task = Task(
            description="Analyze the collected data from Reddit and Twitter, and generate a report detailing customer sentiment, market fit, and product viability.",
            expected_output="A comprehensive market analysis report based on data from Reddit and Twitter, assessing the need and viability of the user's business idea.",
            agent=self.report_agent,
            context=[reddit_search_task, twitter_scraping_task],
        )

        crew = Crew(
            agents=[
                self.controller_agent,
                self.reddit_agent,
                self.twitter_agent,
                self.report_agent,
            ],
            tasks=[
                gather_input_task,
                reddit_search_task,
                twitter_scraping_task,
                generate_report_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    crew = EntrepreneurValidationCrew()
    result = crew.run_crew()
    print(result)
