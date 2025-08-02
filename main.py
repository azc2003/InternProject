from FileParser import FileParser
from TextSearcher import TextSearcher
from chatbot import ChatBot
from comparison_engine import ComparisonEngine
from promptBuilder import PromptBuilder
from datasets import load_dataset

from query_interpreter import QueryInterpreter

if __name__ == "__main__":

    # # 2. 载入 HelpSteer 数据集，只关心 prompt 字段
    # ds = load_dataset("nvidia/HelpSteer", split="train")
    # help_prompts = ds["prompt"]
    # print(f"共加载到 {len(help_prompts)} 条 HelpSteer prompt。")
    #
    # # 3. 看看第一条 Prompt 长什么样
    # print("第一条 HelpSteer prompt：")
    # for i in range(10):
    #     print(help_prompts[i])

    dir_name = "TestFile"
    parser = FileParser(dir_name)
    texts = parser.readDirectory()

    searcher = TextSearcher(model_name="BAAI/bge-large-en-v1.5")
    searcher.build_index(texts)

    # queries= [
    #     "Transformer-in-Transformer (TNT) 架构的核心思想是什么？",  # transformer_in_transformer.pdf
    #     "在教育中使用聊天机器人有哪些主要好处？",  # Article Title.pdf (Role of AI in Education)
    #     "收缩式采矿法的优点是什么？",
    #     # Key-Deposit-Indicators-KDI-and-Key-Mining-Method-Indicators-KMI-in-Underground-Mining-Method-Selection.pdf
    #     "根据 1989 年 NCTM 标准，针对所有学生的五项总体目标是什么？",  # The-math-wars.pdf
    #     "LoRaBlink 协议的设计目标是什么？",  # LoRa for the Internet of Things.pdf
    #     "Reformer 模型引入了哪两种技术来提高 Transformer 的效率？",  # REFORMER_THE_EFFICIENT_TRANSFORMER.pdf
    #     "关于体育教育（PESS）对认知能力的益处，现有研究得出了什么结论？",  # EDUCATIONAL BENEFITS CLAIMED.pdf
    #     "什么是模型上下文协议（MCP）？"  # mcp.txt
    # ]

    queries = [
        # Compare 类
        # "在地下采矿方法选择过程中，关键矿体指标（KDI）与关键采矿方法指标（KMI）有何主要区别？",
        # Key-Deposit-Indicators-KDI-and-Key-Mining-Method-Indicators-KMI-in-Underground-Mining-Method-Selection.pdf
        # "根据《数学战争》一文，传统数学课程与改革导向（标准本位）课程在数学教育理念和实践上有何不同？",  # The-math-wars.pdf
        "人工智能（AI）在教育中与传统教育方式相比有哪些区别？"  # Article Title.pdf

        # Single 类
        # "学术综述中将体育与学校体育（PESS）的声称教育益处分为哪四个主要领域？",  # EDUCATIONAL BENEFITS CLAIMED.pdf
        # "根据《LoRa for the Internet of Things》论文，LoRa收发器有哪些独特特性使其适用于物联网网络？",
        # LoRa for the Internet of Things.pdf
        # "Reformer模型采用了哪些技术来提升Transformer在处理长序列时的效率？"  # REFORMER_THE_EFFICIENT_TRANSFORMER.pdf
    ]

    bot = ChatBot()
    interpreter = QueryInterpreter(bot)
    engine = ComparisonEngine(bot, searcher)
    qa_prompt_builder = PromptBuilder()

    for idx, user_query in enumerate(queries, 1):
        print(f"\n===== Query {idx}: {user_query} =====")
        parsed = interpreter.interpret(user_query)
        print(parsed)
        selected_docs = engine.process(parsed)
        final_prompt = qa_prompt_builder.build_prompt(parsed["translated_input"], selected_docs)
        response = bot.ask(final_prompt)
        print(response)
