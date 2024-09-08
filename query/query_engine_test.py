# -*- coding: utf-8 -*-
# @Time: 2024/9/5 16:43
# @Author: Sui Yuan
# @Software: PyCharm
# @Desc:

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, Document
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from custom.glm_custom_llm import ChatGLM, ZHIPU_API_KEY
from custom.glm_custom_embeding import ChatGLMEmbeddings
import pandas as pd
# chromadb的引入位置很重要，放到前面就出现异常艹
import chromadb

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


def get_doc_full_address_from_csv(path) -> list[Document]:
    data = pd.read_csv(path)
    return [Document(text=f"{row['full_address']}") for i, row in data.iterrows()]


def get_full_address_from_csv(path) -> list[TextNode]:
    data = pd.read_csv(path)
    return [TextNode(text=f"{row['full_address']}") for i, row in data.iterrows()]


if __name__ == '__main__':
    # 获取节点集合
    # nodes = get_full_address_from_csv("../data/beijng_town.csv")

    # 加载索引
    db = chromadb.PersistentClient(path="../storage/chroma_db")
    chroma_collection = db.get_or_create_collection("region_level_4")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index_load = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # 查询提示词
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str="我有一批地址数据，现在我需要根据用户输入的地址文本找到，最匹配的一个，只返回结果不需要其他说明"
                                , query_str="我家在东城区的东华街道上")
    # 定义retriever
    retriever = VectorIndexRetriever(index=index_load, similarity_top_k=2, )
    # 定义synthesizer
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", )
    # 组装query_engine
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
    # 查询
    response = query_engine.query(prompt)
    print(response)
