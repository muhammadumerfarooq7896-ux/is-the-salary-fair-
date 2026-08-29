from typing import Optional, List, Dict
from agents.agent import Agent
from agents.postings import JobPosting, SalaryOpportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from openai import OpenAI
import json


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-4o-mini"  

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.openai = OpenAI()
        self.memory = None
        self.opportunity = None
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_job_postings(self) -> str:
        """
        Run the tool to scan
        """
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No postings found"

    def estimate_fair_salary(self, description: str) -> str:
        """
        Run the tool to estimate a fair salary
        """
        self.log("Autonomous Planning agent is estimating salary via Ensemble Agent")
        estimate = self.ensemble.salary(description)
        return f"The estimated fair salary for {description} is {estimate}"

    def notify_user_of_opportunity(
        self, description: str, advertised_salary: float, estimated_fair_salary: float, url: str
    ) -> Dict:
        """
        Run the tool to notify the user
        """
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
        else:
            self.log("Autonomous Planning agent is notifying user")
            self.messenger.notify(description, advertised_salary, estimated_fair_salary, url)
            posting = JobPosting(job_description=description, salary=advertised_salary, url=url)
            gap = estimated_fair_salary - advertised_salary
            self.opportunity = SalaryOpportunity(
                posting=posting, estimate=estimated_fair_salary, gap=gap
            )
        return "Notification sent ok"

    scan_function = {
        "name": "scan_the_internet_for_job_postings",
        "description": "Returns top job postings scraped from the internet along with the salary each role is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_fair_salary",
        "description": "Given the description of a job posting, estimate what it should fairly pay",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the job posting to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_opportunity",
        "description": "Send the user a push notification about the single most compelling underpaid job posting; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the job posting itself scraped from the internet",
                },
                "advertised_salary": {
                    "type": "number",
                    "description": "The salary offered by this posting as scraped from the internet",
                },
                "estimated_fair_salary": {
                    "type": "number",
                    "description": "The estimated fair salary this role should actually pay",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of this posting as scraped from the internet",
                },
            },
            "required": ["description", "advertised_salary", "estimated_fair_salary", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        """
        Return the json for the tools to be used
        """
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        """
        Actually call the tools associated with this message
        """
        mapping = {
            "scan_the_internet_for_job_postings": self.scan_the_internet_for_job_postings,
            "estimate_fair_salary": self.estimate_fair_salary,
            "notify_user_of_opportunity": self.notify_user_of_opportunity,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    system_message = "You find underpaid job postings using your tools, and notify the user of the best opportunity."
    user_message = """
    First, use your tool to scan the internet for job postings. Then for each posting, use your tool to estimate a fair salary.
    Then pick the single most compelling posting where the fair salary estimate is much higher than what's being offered, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[str] = []) -> Optional[SalaryOpportunity]:
        """
        Run the full workflow, providing the LLM with tools to surface scraped postings to the user
        :param memory: a list of URLs that have been surfaced in the past
        :return: a SalaryOpportunity if one was surfaced, otherwise None
        """
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=self.MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
