import os
import sys
import logging
import json
import threading
from typing import List
from dotenv import load_dotenv
import chromadb
from agents.autonomous_planning_agent import AutonomousPlanningAgent
from agents.postings import SalaryOpportunity
from sklearn.manifold import TSNE
import numpy as np

load_dotenv(override=True)


BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

#
CATEGORIES = [
    "Software Engineering",
    "Data Science",
    "DevOps",
    "Machine Learning",
    "Product Management",
    "Design",
    "Other Tech",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple"]
DEFAULT_COLOR = "gray"  


DB = "salary_vectorstore"  
COLLECTION_NAME = "jobs" 

_chroma_lock = threading.Lock()
_chroma_client = None
_chroma_collections = {}


def get_chroma_client():
    """Return the process-wide chromadb client, creating it once if needed."""
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:

            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(path=DB)
    return _chroma_client


def get_chroma_collection(name: str = COLLECTION_NAME):
    """Return a cached collection handle, creating it once if needed."""
    collection = _chroma_collections.get(name)
    if collection is None:
        client = get_chroma_client()
        with _chroma_lock:
            collection = _chroma_collections.get(name)
            if collection is None:
                collection = client.get_or_create_collection(name)
                _chroma_collections[name] = collection
    return collection


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class JobAgentFramework:
    DB = DB  
    COLLECTION_NAME = COLLECTION_NAME
    MEMORY_FILENAME = "memory.json" 

    def __init__(self):
        init_logging()
        self.memory = self.read_memory()

        self.collection = get_chroma_collection(self.COLLECTION_NAME)
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = AutonomousPlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[SalaryOpportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            opportunities = [SalaryOpportunity(**item) for item in data]
            return opportunities
        return []

    def write_memory(self) -> None:
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
        truncated = data[:2]
        with open(cls.MEMORY_FILENAME, "w") as file:
            json.dump(truncated, file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[SalaryOpportunity]:
        self.init_agents_as_needed()
        logging.info("Kicking off Autonomous Planning Agent")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Autonomous Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):

        collection = get_chroma_collection(cls.COLLECTION_NAME)
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]

        unknown_seen = set()
        colors = []
        for c in categories:
            if c in CATEGORIES:
                colors.append(COLORS[CATEGORIES.index(c)])
            else:
                if c not in unknown_seen:
                    logging.warning(
                        f"Category '{c}' not in CATEGORIES list — using default color. "
                        f"Consider adding it to CATEGORIES in job_agent_framework.py."
                    )
                    unknown_seen.add(c)
                colors.append(DEFAULT_COLOR)
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    JobAgentFramework().run()