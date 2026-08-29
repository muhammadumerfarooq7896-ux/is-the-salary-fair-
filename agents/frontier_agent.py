import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent
import os


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, collection):
        """
        Set up this instance by connecting to OpenAI, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing Frontier Agent")
        # Standard OpenAI client, no custom base_url needed
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.MODEL = "gpt-4o-mini"  # small, cheap OpenAI model - swap for gpt-4.1-mini or o4-mini if preferred
        self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], salaries: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar job postings to the one being estimated
        :param salaries: salaries of the similar postings
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other job postings that might be similar to the role you need to estimate.\n\n"
        for similar, salary in zip(similars, salaries):
            message += f"Potentially related posting:\n{similar}\nSalary is ${salary:.2f}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], salaries: List[float]
    ) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        :param description: a description of the job posting
        :param similars: similar postings to this one
        :param salaries: salaries of similar postings
        :return: the list of messages in the format expected by the API
        """
        message = f"Estimate the fair annual salary in USD for this job. Respond with the salary, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, salaries)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        """
        Return a list of postings similar to the given one by looking in the Chroma datastore
        """
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar postings"
        )
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        salaries = [m["salary"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar postings")
        return documents, salaries

    def get_salary(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def salary(self, description: str) -> float:
        """
        Make a call to OpenAI to estimate the salary of the described job posting,
        by looking up 5 similar postings and including them in the prompt to give context
        :param description: a description of the job posting
        :return: an estimate of the salary
        """
        documents, salaries = self.find_similars(description)
        self.log(
            f"Frontier Agent is about to call {self.MODEL} with context including 5 similar postings"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, salaries),
            seed=42,
        )
        reply = response.choices[0].message.content
        result = self.get_salary(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result