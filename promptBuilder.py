class PromptBuilder:
    def __init__(self, template_path="prompt_template.txt"):
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def build_prompt(self, query: str, docs: list[str]) -> str:
        numbered_docs = "\n".join(
            [
                f"[{i + 1}] (File: {doc['filename']}, Path: {doc['filepath']}, Modified: {doc['modified_time']})\n{doc['text']}"
                for i, doc in enumerate(docs)]
        )
        prompt = self.template.replace("<QUERY>", query).replace("<DOCS>", numbered_docs)
        return prompt


