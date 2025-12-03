import os
import json
import argparse
from tqdm import tqdm
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from rpm_mcts_tools.utils.utils import read_json, write_jsonl


def state_action_build(file_path, output_path):
    data = read_json(file_path)
    sa_pair = []
    for item in tqdm(data[:]):
        problem = item['prompt']
        canonical_steps = item['canonical_steps']
        topic = item['topic']
        pre_steps = problem
        for idx, step in enumerate(canonical_steps, start=1):
            pre_steps += "\n" + step
            sa_pair.append({"pre_steps": pre_steps, "topic": topic, "pre_steps_num": idx})
    with open(output_path, "w") as f:
        json.dump(sa_pair, f, indent=4)


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata.update(record)
    return metadata


def vector_cache_build(action_pair, vector_cache_path, persist_directory, emb_model_name, content_key):
    print("vector cache begin building...")
    embedding = HuggingFaceEmbeddings(
        model_name=f"../../huggingface/{emb_model_name}",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    loader = JSONLoader(
        file_path=action_pair,
        jq_schema=".[] ",
        content_key=content_key,
        metadata_func=metadata_func,
    )
    documents = loader.load()
    print(len(documents))

    store = LocalFileStore(os.path.abspath(vector_cache_path))
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding,
        store,
        namespace=emb_model_name
    )
    db = Chroma.from_documents(
        documents, 
        cached_embedder,
        persist_directory=persist_directory,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    print("Success Build BGE")


def main(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.output_dir, exist_ok=True)

    # construct output paths
    args.document_path = os.path.join(args.output_dir, "document.json")
    args.vector_cache_path = os.path.join(args.output_dir, "vectors")
    args.persist_directory = os.path.join(args.output_dir, "chroma_db")

    # make document pairs
    state_action_build(args.state_file_path, args.document_path)
    
    # embedding and build vector cache
    vector_cache_build(args.document_path, args.vector_cache_path, args.persist_directory, args.emb_model_name, args.content_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_file_path", type=str, default="../../output/canonical_steps/processed_canonical_steps_with_topic.jsonl")
    parser.add_argument("--output_dir", type=str, default="../../output/knowledge_base/kb_2_v3")
    parser.add_argument("--emb_model_name", type=str, default="bge-large-en-v1.5")
    parser.add_argument("--content_key", type=str, default="pre_steps")
    args = parser.parse_args()
    main(args)
