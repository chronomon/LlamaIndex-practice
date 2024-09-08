# -*- coding: utf-8 -*-
# @Time: 2024/9/3 20:15
# @Author: Sui Yuan
# @Software: PyCharm
# @Desc:
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.prompts import PromptTemplate
from custom.glm_custom_llm import ChatGLM, ZHIPU_API_KEY
from custom.glm_custom_embeding import ChatGLMEmbeddings
import pandas as pd

# define our LLM
Settings.llm = ChatGLM(model='glm-4', reuse_client=True, api_key=ZHIPU_API_KEY, )
# define embed model
Settings.embed_model = ChatGLMEmbeddings(model='embedding-2', reuse_client=True, api_key=ZHIPU_API_KEY, )
# 构建prompt
template = (
    "我在下方提供一下上下文背景.\n"
    "------------------\n"
    "{context_str}"
    "\n----------------\n"
    "基于上面的背景知识，请回答问题：{query_str}\n")


class RegionSplitter:
    def get_nodes_from_documents(self, documents):
        nodes = []
        for doc in documents:
            regions = doc.text.split('\n')
            for region in regions:
                props = region.split(',')
                nodes.append(TextNode(text=props[2]))
        return nodes


def get_full_address_from_loader(path) -> list[BaseNode]:
    documents = SimpleDirectoryReader(input_files=[path]).load_data()
    parser = RegionSplitter()
    return parser.get_nodes_from_documents(documents)


def get_full_address_from_csv(path) -> list[TextNode]:
    data = pd.read_csv(path)
    return [TextNode(text=f"{row['full_address']}") for i, row in data.iterrows()]


if __name__ == '__main__':
    qa_template = PromptTemplate(template)

    # 获取节点集合
    nodes = get_full_address_from_loader("../data/beijng_town.csv")
    # 定义索引
    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    print(index.ref_doc_info)
    # 查询提示词
    prompt = qa_template.format(context_str="我有一批地址数据，现在我需要根据用户输入的地址文本找到，最匹配的一个，只返回结果不需要其他说明"
                                , query_str="我家在东城区的东华街道上")
    # 查询引擎
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    print(response)
