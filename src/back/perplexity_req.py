from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class PerplexityAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it as parameter or PERPLEXITY_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")
        self.model = "sonar"
        self.system_role = {
            "role": "system",
            "content": (
                "You are an alert notification system and must provide responses with "
                "the following structure: '[Risk]. [Illness] in [Country]. [Current date]'"
            )
        }
    
    def ask(self, query):
        message = [
            self.system_role,
            {
                "role": "user",
                "content": (
                    query
                )
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
        )
        return response
