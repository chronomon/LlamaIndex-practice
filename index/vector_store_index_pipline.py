# -*- coding: utf-8 -*-
# @Time: 2024/9/3 20:15
# @Author: Sui Yuan
# @Software: PyCharm
# @Desc:

from typing import Any, Dict, List, Optional, Sequence, Type
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.node_parser import NodeParser, TokenTextSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor
from llama_index.core.utils import get_tqdm_iterable
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


class RegionSplitter(NodeParser):

    def get_nodes_from_documents(self, documents, **kwargs):
        nodes = []
        for doc in documents:
            regions = doc.text.split('\n')
            for region in regions:
                props = region.split(',')
                nodes.append(TextNode(text=props[2]))
        return nodes

    def _parse_nodes(
            self,
            nodes: Sequence[BaseNode],
            show_progress: bool = False,
            **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        documents_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing documents into nodes"
        )
        for document in documents_with_progress:
            nodes = self.get_nodes_from_documents([document], show_progress)
            all_nodes.extend(nodes)
        return all_nodes


if __name__ == '__main__':
    # 用pipline初始化数据
    documents = SimpleDirectoryReader(input_files=["../data/beijng_town.csv"]).load_data()
    pipline = IngestionPipeline(
        transformations=[
            RegionSplitter(),
            TitleExtractor()]
    )
    nodes = pipline.run(documents=documents)

    # 定义索引
    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    # 查询提示词
    qa_template = PromptTemplate(template)

    prompt = qa_template.format(context_str="我有一批地址数据，现在我需要根据用户输入的地址文本找到，最匹配的一个，只返回结果不需要其他说明"
                                , query_str="我家在东城区的东华街道上")
    # 查询引擎
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    print(response)
