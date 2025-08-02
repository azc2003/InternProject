import json
import re
from chatbot import ChatBot

class QueryInterpreter:
    def __init__(self, chatbot: ChatBot, template_path="PromptTemplateFile/query_prompt_template.txt"):
        self.bot = chatbot
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def interpret(self, user_input: str) -> dict:

        prompt = self.template.replace("<USER_INPUT>", user_input)
        response = self.bot.ask(prompt)

        cleaned = response.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM, cleaned response was:\n{cleaned!r}")
