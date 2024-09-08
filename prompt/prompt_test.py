# -*- coding: utf-8 -*-
# @Time: 2024/8/29 20:31
# @Author: Sui Yuan
# @Software: PyCharm
# @Desc:
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from custom.glm_custom_llm import ChatGLM, ZHIPU_API_KEY

# define our LLM
Settings.llm = ChatGLM(model='GLM-4-0520', reuse_client=True, api_key=ZHIPU_API_KEY, )

# define embed model
# Settings.embed_model = ChatGLMEmbeddings(model='embedding-2', reuse_client=True, api_key=ZHIPU_API_KEY, )
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
# Load the your data
documents = SimpleDirectoryReader(input_files=["./data/position"]).load_data()

# 构建prompt
template = (
    "我在下方提供一下上下文背景.\n"
    "------------------\n"
    "{context_str}"
    "\n----------------\n"
    "基于上面的背景知识，请回答问题：{query_str}\n")

qa_template = PromptTemplate(template)
prompt = qa_template.format(context_str="税务BP换人了", query_str="税务BP是谁？")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine()
response = query_engine.query(prompt)
print(response)
