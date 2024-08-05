import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Based on the following context and your own knowledge, answer the question:

Context:
{context}

Question: {question}

Your answer should consider both the specific context provided and relevant information you know.
"""
GPT_ONLY_PROMPT_TEMPLATE = """
Based on your own knowledge, answer the question:

Question: {question}

Your answer should consider relevant information you know.
"""

def main():
    # Create CLI.
    query_text = input("Please enter your query: ")


    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key="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")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
     # Response based on database context
    model = ChatOpenAI(openai_api_key="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")
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    document = [doc.metadata.get("start_index", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}\nStart_Index: {document}\n"
    print(formatted_response)
    # Response based on GPT-3 knowledge only
    gpt_only_prompt = GPT_ONLY_PROMPT_TEMPLATE.format(question=query_text)
    gpt_based_response = model.predict(gpt_only_prompt)
    print("\nResponse based on GPT-3 knowledge only:")
    print(gpt_based_response)

if __name__ == "__main__":
    main()
