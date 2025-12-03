import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def filter_similar_sentences(sentences, embedding_model, similarity_threshold=0.85):
    vectors = embedding_model.embed_documents(sentences)
    vectors = np.array(vectors)

    # 用于存储要保留的句子的索引
    retained_indices = []
    for i in range(len(sentences)):
        is_similar = False
        for j in retained_indices:
            # 计算余弦相似度，如果大于阈值，则不保留
            similarity = cosine_similarity(vectors[i], vectors[j])
            print(f"Similarity between sentence {i} and sentence {j}: {similarity}")
            if similarity > similarity_threshold:
                is_similar = True
                break
        if not is_similar:
            retained_indices.append(i)

    # 根据保留的索引提取句子
    filtered_sentences = [sentences[i] for i in retained_indices]
    if len(filtered_sentences) < len(sentences):
        print(f"因相似度超过阈值{similarity_threshold}过滤掉的句子数量: {len(sentences) - len(filtered_sentences)}")
    return filtered_sentences


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(
        model_name="../huggingface/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    texts = [
        "Step 1: We can use the find() method to find the index of the first occurrence of the character and the rfind() method to find the index of the last occurrence.",
        "Step 1: We can use the built - in string method 'find' to find the index of the first occurrence of the character and 'rfind' to find the index of the last occurrence.",
        "Step 1: Use the string method to iterate through the string and find the position of the first occurrence of the character 'ch' in the string 's'."
    ]

    # texts = [
    #     "Step 2: Use the string method to iterate through the string in reverse order and find the position of the last occurrence of the character 'ch' in the string 's'.",
    #     "Step 2: Use the string method to iterate through the string in reverse and find the position of the last occurrence of the character 'ch' in the string 's'.",
    #     "Step 2: Use the string method to iterate through the string in reverse and find the position of the last occurrence of the character 'ch' in the string 's'."
    # ]

    # texts = [
    #     "Step 3: We can use string slicing. If the first index is 'f' and the last index is 'l', the new string can be created as s[:f] + s[f+1:l] + s[l+1:].",
    #     "Step 3: To create the new string, we can use string slicing. If the index of the first occurrence is 'first_index' and the index of the last occurrence is 'last_index', the new string can be created as s[:first_index] + s[first_index+1:last_index] + s[last_index+1:].",
    #     "Step 3: Let's assume the index of the first occurrence is 'first_index' and the index of the last occurrence is 'last_index'. Then the new string can be created as s[:first_index] + s[first_index+1:last_index] + s[last_index+1:]."
    # ]

    filtered_texts = filter_similar_sentences(texts, embedding_model)
    print(filtered_texts)