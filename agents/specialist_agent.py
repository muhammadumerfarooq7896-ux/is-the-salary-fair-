import modal
from agents.agent import Agent

class SalarySpecialistAgent(Agent):
    """
    An Agent that runs our fine-tuned LLM that's running remotely on Modal
    """

    name = "Salary Specialist Agent"
    color = Agent.RED
    def __init__(self):
        """
        Set up this Agent by connecting to the deployed Modal function
        """
        self.log("Salary Specialist Agent is initializing - connecting to modal")
        self.salary_model=modal.Function.from_name("salary-service","price")
    
    def salary(self,description):
        """
        Make a remote call to return the estimate of the salary for this job
        """
        self.log("Salary Specialist Agent is calling remote fine-tuned model")
        result=self.salary_model.remote(description)
        self.log(f"Salary Specialist Agent completed - predicting ${result:,.2f}")
        return result
    