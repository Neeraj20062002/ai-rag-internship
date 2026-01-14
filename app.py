import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (kept for future use)
load_dotenv()

DATA_PATH = "data"
INDEX_PATH = "faiss_index"

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def retrieve_context(query, k=3):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_answer(context, query):
    """
    Simple answer synthesis.
    This simulates an LLM response using retrieved context.
    """
    answer = f"""
Answer based on retrieved context:

{context}

Question:
{query}
"""
    return answer.strip()

if __name__ == "__main__":
    query = input("Ask a question: ")

    context = retrieve_context(query)
    answer = generate_answer(context, query)

    print("\n--- RAG RESPONSE ---\n")
    print(answer)
