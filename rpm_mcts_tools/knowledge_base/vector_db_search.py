import os
import time
import json
import argparse
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# 因精度问题query相等时可能会出现相似度分数大于1的情况，镇压警告
import warnings
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

class DocumentSearch:
    def __init__(self, persist_directory, embedding_model):
        # 检查持久化目录是否存在
        if not os.path.exists(persist_directory):
            print(f"Error: The persist directory {persist_directory} does not exist.")
            exit(1)

        # 初始化database
        self.embedding = embedding_model
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            # collection_configuration={"hnsw": {"space": "cosine"}},   # 注意：在这里设置向量距离计算方式无效，必须在创建数据库时设置
        )

    def search_by_query_with_relevance_scores(self, query, k=4, score_threshold=0.0, filter=None):
        """
        Return docs and relevance scores in the range [0, 1].
        0 is dissimilar, 1 is most similar.
        """
        kwargs = {
            "score_threshold": score_threshold,
            "filter": filter,
        }
        search_results = self.db.similarity_search_with_relevance_scores(query, k=k, **kwargs)
        # 将搜索结果转为字典列表
        final_results = [{**doc.metadata, "similarity": score} for doc, score in search_results]
        return final_results

def main(args):
    # test case
    query = "def similar_elements(test_tup1, test_tup2):\n    \"\"\"\n    Write a function to find the shared elements from the given two lists.\n    \"\"\"\n"
    print("query:", query)
    
    # 创建DocumentSearch
    embedding_model = HuggingFaceEmbeddings(
        model_name=f"../../huggingface/{args.emb_model_name}",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    document_search = DocumentSearch(
        persist_directory=os.path.join(args.db_path, "chroma_db"),
        embedding_model=embedding_model
    )
    
    # 检索
    start_time = time.time()
    search_results = document_search.search_by_query_with_relevance_scores(query, k=3)
    print("检索结果:")
    for i, result in enumerate(search_results):
        print('-'*50)
        print(f"Result {i+1}:")
        print(json.dumps(result, indent=4, ensure_ascii=False, sort_keys=True))
    print(f"检索耗时: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_model_name", type=str, default="bge-large-en-v1.5")
    parser.add_argument("--db_path", type=str, default="../../output/knowledge_base/test")
    args = parser.parse_args()
    main(args)
