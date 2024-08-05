from langchain.evaluation import load_evaluator
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama

def main():
    # Get embedding for a word.
    local_embedding_function = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
    evaluator = load_evaluator("pairwise_embedding_distance",embeddings=local_embedding_function)
    words = ("我很高興", "我很難過")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()