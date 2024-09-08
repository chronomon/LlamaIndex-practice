# -*- coding: utf-8 -*-
# @Time: 2024/9/6 13:25
# @Author: Sui Yuan
# @Software: PyCharm
# @Desc:

from llama_index.core import Settings
from llama_index.core import SimpleKeywordTableIndex, KeywordTableIndex
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from custom.glm_custom_llm import ChatGLM, ZHIPU_API_KEY
from custom.glm_custom_embeding import ChatGLMEmbeddings
import pandas as pd
# chromadb的引入位置很重要，放到前面就出现异常艹
import chromadb

# define our LLM
Settings.llm = ChatGLM(model='glm-4', reuse_client=True, api_key=ZHIPU_API_KEY, )

# define embed model
Settings.embed_model = ChatGLMEmbeddings(model='embedding-2', reuse_client=True, api_key=ZHIPU_API_KEY, )

def get_full_address_from_csv(path) -> list[TextNode]:
    data = pd.read_csv(path)
    return [TextNode(text=f"{row['full_address']}") for i, row in data.iterrows()]


def get_simple_table_retriever(nodes):
    # 定义索引
    # SimpleKeywordTableIndex会根据r"\w+"来提取所有连续的字母和汉字以及下划线，因此，如果query是一段纯文字，倒排索引会匹配不上
    index = SimpleKeywordTableIndex(nodes=nodes, show_progress=True)

    # 定义retriever
    retriever = index.as_retriever(retriever_mode="simple")
    return retriever


def get_GPT_table_retriever(nodes):
    # 定义索引
    # KeywordTableIndex会基于promptTemplate用llm来进行分词(如果不设置prompt会使用以下系统默认的：)
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"

    keyword_template = "我会在下面提供一段文本，提取出最多{max_keywords}个关键字，并剔除停用词。" \
                       "---------------------\n" \
                       "{text}\n" \
                       "---------------------\n" \
                       "以下面的形式提供关键字结果，不需要其他说明：KEYWORDS: <关键词>"
    prompt_template = PromptTemplate(keyword_template)
    index = KeywordTableIndex(nodes=nodes, keyword_extract_template=prompt_template, show_progress=True)

    # 定义retriever
    retriever = index.as_retriever(retriever_mode="default")
    return retriever


if __name__ == '__main__':
    # 获取节点集合
    nodes = get_full_address_from_csv("../data/beijng_town.csv")

    # 加载索引，定义retriever
    # retriever = get_simple_table_retriever(nodes)
    retriever = get_GPT_table_retriever(nodes)
    res_nodes = retriever.retrieve("我家在东城区的东华街道上")
    for node_score in res_nodes:
        print(f'{node_score.get_content()}|{node_score.get_score()}')
