import os
from agents.postings import SalaryOpportunity
from agents.agent import Agent
from litellm import completion
import requests

pushover_url = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "deepseek/deepseek-chat"  # litellm routes this via DEEPSEEK_API_KEY

    def __init__(self):
        """
        Set up this object to either do push notifications via Pushover,
        or SMS via Twilio,
        whichever is specified in the constants
        """
        self.log("Messaging Agent is initializing")
        self.pushover_user = os.getenv("PUSHOVER_USER", "your-pushover-user-if-not-using-env")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "your-pushover-user-if-not-using-env")
        self.log("Messaging Agent has initialized Pushover and DeepSeek")

    def push(self, text):
        """
        Send a Push Notification using the Pushover API
        """
        self.log("Messaging Agent is sending a push notification")
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        requests.post(pushover_url, data=payload)

    def alert(self, opportunity: SalaryOpportunity):
        """
        Make an alert about the specified SalaryOpportunity
        """
        text = f"Salary Alert! Advertised=${opportunity.posting.salary:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Gap=${opportunity.gap:.2f} :"
        text += opportunity.posting.job_description[:10] + "... "
        text += opportunity.posting.url
        self.push(text)
        self.log("Messaging Agent has completed")

    def craft_message(
        self, description: str, advertised_salary: float, estimated_fair_salary: float
    ) -> str:
        gap = estimated_fair_salary - advertised_salary

        user_prompt = (
            "This job posting is UNDERPAID: our fair-salary model estimates it should pay "
            f"significantly more than what's advertised (a gap of about ${gap:,.0f}). "
            "Write a 2-3 sentence push notification that clearly conveys this is a pay-gap "
            "opportunity — state the advertised salary, the estimated fair salary, and the "
            "gap between them, then briefly describe the role so the user knows what it is. "
            "Tone: alert/exciting, but the pay mismatch must be the headline, not an afterthought.\n"
        )
        user_prompt += f"Job Description: {description}\nAdvertised Salary: {advertised_salary}\nEstimated Fair Salary: {estimated_fair_salary}"
        user_prompt += "\n\nRespond only with the 2-3 sentence message itself, no preamble."
        response = completion(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def notify(self, description: str, advertised_salary: float, estimated_fair_salary: float, url: str):
        """
        Make an alert about the specified details
        """
        self.log("Messaging Agent is using DeepSeek to craft the message")
        text = self.craft_message(description, advertised_salary, estimated_fair_salary)
        self.push(text[:200] + "... " + url)
        self.log("Messaging Agent has completed")