from openai import OpenAI

class ChatBot:
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1", model="openai/gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def ask(self, message: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
