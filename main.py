from TextSearcher import TextSearcher
from chatbot import ChatBot
from pdfparser import PDFTextParser
from vectorizer import Vectorizer

if __name__ == "__main__":
    pdf_file = "manuel.pdf"
    parser = PDFTextParser(pdf_file)
    texts = parser.extract_blocks()

    searcher = TextSearcher(model_name="all-MiniLM-L6-v2")
    searcher.build_index(texts)

    query = "Medical Device Interference"
    results = searcher.search(query, k=3)

    api_key = "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"
    bot = ChatBot(api_key=api_key)
    llm_prompt = results[0]["text"]
    print(bot.ask(llm_prompt))




