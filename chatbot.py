from openai import OpenAI
import config

class ChatBot:
    def __init__(self, api_key = config.OPENAI_API_KEY, base_url= config.OPENAI_BASE_URL, model= config.OPENAI_MODEL):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def ask(self, message: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
