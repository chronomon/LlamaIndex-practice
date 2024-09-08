# -*- coding: utf-8 -*-
# @Time: 2024/9/4 15:53
# @Author: Sui Yuan
# @Software: PyCharm
# @Desc:

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, Document
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from custom.glm_custom_llm import ChatGLM, ZHIPU_API_KEY
from custom.glm_custom_embeding import ChatGLMEmbeddings
import pandas as pd
# chromadb的引入位置很重要，放到前面就出现异常fuck
import chromadb

# define our LLM
Settings.llm = ChatGLM(model='glm-4', reuse_client=True, api_key=ZHIPU_API_KEY, )

# define embed model
Settings.embed_model = ChatGLMEmbeddings(model='embedding-2', reuse_client=True, api_key=ZHIPU_API_KEY, )
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

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
    nodes = get_full_address_from_csv("../data/beijng_town.csv")
    # documents = get_doc_full_address_from_csv("../data/beijng_town.csv")
    # documents = SimpleDirectoryReader(input_files=["../data/position"]).load_data()

    # 定义、存储index
    db = chromadb.PersistentClient(path="./chroma_db")
    # db = chromadb.EphemeralClient()
    chroma_collection = db.get_or_create_collection("region_level_4")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # 存储索引
    # index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)
    # index = VectorStoreIndex(nodes=nodes, show_progress=True, storage_context=storage_context)
    # 加载索引
    index_load = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # 查询提示词
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str="我有一批地址数据，现在我需要根据用户输入的地址文本找到，最匹配的一个，只返回结果不需要其他说明"
                                , query_str="我家在东城区的东华街道上")
    # 查询引擎
    query_engine = index_load.as_query_engine()
    response = query_engine.query(prompt)
    print(response)
