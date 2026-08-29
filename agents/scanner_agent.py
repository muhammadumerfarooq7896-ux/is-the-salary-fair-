from typing import Optional, List
from openai import OpenAI
from agents.postings import ScrapedPosting, PostingSelection
from agents.agent import Agent


class ScannerAgent(Agent):

    MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed job postings from a list, by selecting postings that have the most detailed, high quality description and the most clear salary information.
    Respond strictly in JSON with no explanation, using this format. You should provide the salary as a number derived from the description. If the salary of a posting isn't clear, do not include that posting in your response.
    Most important is that you respond with the 5 postings that have the most detailed job description with salary. It's not important to mention application instructions or company perks; most important is a thorough description of the role, responsibilities, and requirements.
    Be careful with postings that describe a salary range vaguely (e.g. "competitive salary") — only respond with postings when you are highly confident about the actual salary figure.
    """

    USER_PROMPT_PREFIX = """Respond with the most promising 5 job postings from this list, selecting those which have the most detailed, high quality description and a clear salary that is greater than 0.
    You should rephrase the description to be a summary of the role itself — responsibilities, seniority, required skills — not application instructions or company boilerplate.
    Remember to respond with a short paragraph of text in the job_description field for each of the 5 postings that you select.
    Be careful with vague salary language like "competitive" or "DOE" — only respond with postings when you are highly confident about the actual salary figure.

    Postings:

    """

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 postings, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Set up this instance by initializing OpenAI
        """
        self.log("Scanner Agent is initializing")
        self.openai = OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_postings(self, memory) -> List[ScrapedPosting]:
        """
        Look up postings published via the Jobicy API
        Return any new postings that are not already in the memory provided
        """
        self.log("Scanner Agent is about to fetch postings from Jobicy")
        urls = [opp.posting.url for opp in memory]
        scraped = ScrapedPosting.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} postings not already scraped")
        return result

    def make_user_prompt(self, scraped) -> str:
        """
        Create a user prompt for the model based on the scraped postings provided
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str] = []) -> Optional[PostingSelection]:
        """
        Call the model to provide a high potential list of postings with good descriptions and salaries
        :param memory: a list of URLs representing postings already raised
        :return: a selection of good postings, or None if there aren't any
        """
        scraped = self.fetch_postings(memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log(f"Scanner Agent is calling {self.MODEL} using Structured Outputs")
            result = self.openai.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=PostingSelection,
            )
            selection = result.choices[0].message.parsed
            selection.postings = [p for p in selection.postings if p.salary > 0]
            self.log(
                f"Scanner Agent received {len(selection.postings)} selected postings with salary>0 from OpenAI"
            )
            return selection
        return None

    def test_scan(self, memory: List[str] = []) -> Optional[PostingSelection]:
        """
        Return a test PostingSelection, to be used during testing without burning
        real Jobicy/Groq calls
        """
        results = {
            "postings": [
                {
                    "job_description": "Senior Backend Engineer role focused on building scalable Python microservices for a fintech platform. Requires 5+ years experience with distributed systems, PostgreSQL, and AWS. Leads code reviews and mentors junior engineers.",
                    "salary": 165000,
                    "url": "https://jobicy.com/jobs/example-backend-engineer-1",
                },
                {
                    "job_description": "Mid-level Data Scientist position working on churn prediction models for a subscription SaaS company. Requires 3+ years experience with Python, scikit-learn, and SQL. Collaborates closely with the product team.",
                    "salary": 118000,
                    "url": "https://jobicy.com/jobs/example-data-scientist-1",
                },
                {
                    "job_description": "DevOps Engineer responsible for CI/CD pipelines and Kubernetes infrastructure at a mid-size e-commerce company. Requires 4+ years experience with Terraform, Docker, and AWS/GCP.",
                    "salary": 135000,
                    "url": "https://jobicy.com/jobs/example-devops-1",
                },
                {
                    "job_description": "Junior Frontend Developer building React-based dashboards for an internal analytics tool. Requires 1-2 years experience with React, TypeScript, and REST APIs.",
                    "salary": 78000,
                    "url": "https://jobicy.com/jobs/example-frontend-1",
                },
                {
                    "job_description": "Staff Machine Learning Engineer leading the design of a recommendation system serving millions of users. Requires 8+ years experience with production ML systems, PyTorch, and large-scale data pipelines.",
                    "salary": 210000,
                    "url": "https://jobicy.com/jobs/example-ml-staff-1",
                },
            ]
        }
        return PostingSelection(**results)
