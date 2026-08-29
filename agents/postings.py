from pydantic import BaseModel, Field
from typing import List, Dict, Self
import requests
import time



JOBICY_ENDPOINT = "https://jobicy.com/api/v2/remote-jobs"



QUERIES = [
    {"count": 50, "industry": "engineering"},
    {"count": 50, "industry": "data-science"},
    {"count": 50, "industry": "admin"},
]


class ScrapedPosting:
    """
    A class to represent a job posting retrieved from the Jobicy API

    """

    title: str
    company: str
    location: str
    description: str
    url: str
    salary_min: float | None
    salary_max: float | None

    def __init__(self, entry: Dict):
        """
        Populate this instance from one job object in Jobicy's JSON response.
        Field names below (jobTitle, companyName, etc.) follow Jobicy's documented
        schema
        """
        self.title = entry.get("jobTitle", "")[:150]
        self.company = entry.get("companyName", "")
        self.location = entry.get("jobGeo", "") or "Remote"
        self.description = (entry.get("jobDescription") or entry.get("jobExcerpt") or "")[:3000]
        self.url = entry.get("url", "")
        self.salary_min = entry.get("annualSalaryMin")
        self.salary_max = entry.get("annualSalaryMax")

    def __repr__(self):
        return f"<{self.title} @ {self.company}>"

    def describe(self):
        """
        Return a string to describe this posting for use in calling a model
        (direct equivalent of ScrapedDeal.describe())
        """
        return f"Title: {self.title}\nCompany: {self.company}\nLocation: {self.location}\nDescription: {self.description.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve postings from Jobicy across the configured queries

        """
        postings = []
        seen_urls = set()
        query_iter = QUERIES
        for params in query_iter:
            try:
                resp = requests.get(JOBICY_ENDPOINT, params=params)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                # Don't let one bad/rejected query (e.g. an invalid industry slug)
                # take down the whole scan — log and move on to the next query.
                import logging
                logging.warning(f"Jobicy query {params} failed: {e}")
                continue
            for entry in data.get("jobs", []):
                posting = cls(entry)
                if posting.url and posting.url not in seen_urls:
                    seen_urls.add(posting.url)
                    postings.append(posting)
            time.sleep(0.2)  
        return postings


class JobPosting(BaseModel):
    """
    A class to represent a JobPosting with salary
    (direct equivalent of Deal, where price -> salary)
    """

    job_description: str = Field(
        description="Your clearly expressed summary of the role in 3-4 sentences. Focus on responsibilities, seniority, and required skills — not company perks or application instructions."
    )
    salary: float = Field(
        description="The estimated or advertised annual salary in USD for this role. If a range is given, use the midpoint."
    )
    url: str = Field(description="The URL of the posting, as provided in the input")


class PostingSelection(BaseModel):
    """
    A class to represent a list of JobPostings
    (direct equivalent of DealSelection)
    """

    postings: List[JobPosting] = Field(
        description="Your selection of the 5 postings that have the most detailed, high quality description and the clearest salary information."
    )


class SalaryOpportunity(BaseModel):
    """
    A class to represent a possible opportunity: a JobPosting where our model
    estimates a fair salary differs meaningfully from what's advertised
    (direct equivalent of Opportunity)
    """

    posting: JobPosting
    estimate: float
    gap: float