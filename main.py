from FileParser import FileParser
from TextSearcher import TextSearcher
from chatbot import ChatBot
from promptBuilder import PromptBuilder
from vectorizer import Vectorizer

if __name__ == "__main__":
    dir_name = "FileLibrary"
    parser = FileParser(dir_name)
    texts = parser.readDirectory()

    searcher = TextSearcher()
    searcher.build_index(texts)

    query = "List all files modified in July 2025."
    results = searcher.search(query, k=5)

    bot = ChatBot()
    promptBuilder = PromptBuilder()
    responsePrompt = promptBuilder.build_prompt(query,results)
    print(bot.ask(responsePrompt))




