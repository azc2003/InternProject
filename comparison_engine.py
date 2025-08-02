import json

from promptBuilder import PromptBuilder
from chatbot import ChatBot
from TextSearcher import TextSearcher

class ComparisonEngine:
    def __init__(self, chatbot: ChatBot, searcher: TextSearcher,
                 compare_template_path: str = "PromptTemplateFile/compare_prompt_template.txt"):

        self.bot = chatbot
        self.searcher = searcher
        self.compare_prompt_builder = PromptBuilder(template_path=compare_template_path)

    def clean_llm_output(self, text: str) -> str:
        text = text.strip()

        if text.lower().startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]

        if text.endswith('```'):
            text = text[:-3]

        return text.strip()

    def format_docs_for_prompt(self, docs: list[dict]) -> str:

        return "\n\n".join([
            f"[{i + 1}] (File: {doc['filename']}, Path: {doc['filepath']}, Modified: {doc['modified_time']})\n{doc['text']}"
            for i, doc in enumerate(docs)
        ])

    def process(self, parsed_query: dict, top_k_per_keyword: int = 5) -> str:

        query_type = parsed_query.get("query_type")
        query = parsed_query.get("translated_input")
        keywords = parsed_query.get("keywords")

        if query_type == "unknown" or not query or not keywords:
            print("Invalid or unknown query.")
            return "No relevant content found."

        if query_type == "single":
            results = self.searcher.search(query, k=top_k_per_keyword)
            return self.format_docs_for_prompt(results)

        elif query_type == "comparison":
            all_results = []
            for keyword in keywords:
                keyword_results = self.searcher.search(keyword, k=top_k_per_keyword)
                all_results.extend(keyword_results)

            prompt = self.compare_prompt_builder.build_prompt(query, self.format_docs_for_prompt(all_results))
            response = self.bot.ask(prompt)
            clean_response = self.clean_llm_output(response)

            result_json = json.loads(clean_response)
            documents_list = result_json.get("top_documents", [])
            documents_string = "\n\n".join(documents_list)

            return documents_string



