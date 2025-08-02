class PromptBuilder:
    def __init__(self, template_path="PromptTemplateFile/prompt_template.txt"):
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def build_prompt(self, query: str, docs: str) -> str:
        prompt = self.template.replace("<QUERY>", query).replace("<DOCS>", docs)
        return prompt


