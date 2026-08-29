from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self

PREFIX = "Salary is $"
QUESTION = "What is the estimated annual salary in USD for this role?"
# -------------------------------------------------------------------------


class JobItem(BaseModel):
    """
    A JobItem is a data-point of a Job Posting with a Salary
    (direct equivalent of Item, where price -> salary)
    """

    title: str
    category: str          # e.g. industry / role family, like his "category"
    salary: float           # equivalent of his "price"
    full: Optional[str] = None       # raw scraped/loaded text before cleanup
    weight: Optional[float] = None
    summary: Optional[str] = None    # cleaned description used in the prompt
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def count_tokens(self, tokenizer) -> int:
        return len(tokenizer.encode(self.summary))

    def count_prompt_tokens(self, tokenizer) -> int:
        return len(tokenizer.encode(self.prompt))

    def make_prompts(self, tokenizer, cutoff: int, include_completion: bool):
        """
        Build prompt (+ completion) from the summary, truncated to `cutoff` tokens.
        Mirrors his make_prompts(tokenizer, CUTOFF, True/False) signature.
        """
        tokens = tokenizer.encode(self.summary)[:cutoff]
        text = tokenizer.decode(tokens)
        if include_completion:
            # Train-time: prompt+completion in one string, model learns to predict the number
            self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.salary)}.00"
            self.completion = f"{PREFIX}{round(self.salary)}.00"
        else:
            # Test-time: withhold the actual number
            self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"
            self.completion = f"{PREFIX}{round(self.salary)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.salary:,.0f}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )

    @staticmethod
    def push_prompts_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        DatasetDict(
            {
                "train": Dataset.from_list([{"prompt": i.prompt, "completion": i.completion} for i in train]),
                "validation": Dataset.from_list([{"prompt": i.prompt, "completion": i.completion} for i in val]),
                "test": Dataset.from_list([{"prompt": i.prompt, "completion": i.completion} for i in test]),
            }
        ).push_to_hub(dataset_name)
