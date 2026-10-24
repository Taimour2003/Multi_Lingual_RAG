from display_data import display_data
from document_processing import document_processing
from Evaluation import context_precision_evaluation
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# from sentence_transformers import SentenceTransformer

# # # document_processing


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

if not Chroma.exists(persist_directory="./vector_store2"):
    chunks_english = document_processing("./IELTS_Prepration.docx")
    chunks_urdu = document_processing("./IELTS_Preparation_Guide_Urdu.docx")
    chunks_arabic = document_processing("./IELTS_Preparation_Guide_Arabic.docx")

    total_chunks = chunks_english + chunks_urdu + chunks_arabic
    vector_store = Chroma.from_documents(
        documents=total_chunks,
        embedding=embeddings,
        persist_directory="./vector_store2",
    )


vector_store = Chroma(
    persist_directory="./vector_store2", embedding_function=embeddings
)
retrievar = vector_store.as_retriever(search_kwargs={"k": 2})
db_data = vector_store.get(include=["documents", "metadatas"])
all_docs = [
    Document(page_content=text, metadata=meta if meta is not None else {})
    for text, meta in zip(db_data["documents"], db_data["metadatas"])
]
BM25_retriever = BM25Retriever.from_documents(all_docs, k=10)
ensembel_retriever = EnsembleRetriever(
    retrievers=[retrievar, BM25_retriever], weights=[0.5, 0.5]
)
# 2. Load the BGE Multilingual Reranker v2 M3
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
# Wrap it in a CrossEncoderReranker compressor to return the top 2 best-scored chunks
compressor = CrossEncoderReranker(model=reranker_model, top_n=2)

# 3. Combine Base Retriever with Reranker
final_reranked_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensembel_retriever
)

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1)


template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

CRITICAL INSTRUCTION: You must respond in the exact same language that the user asked the question in. If the question is in Urdu, answer entirely in Urdu. If the question is in Arabic, answer entirely in Arabic. If the question is in English, answer in English.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


rag_chain = (
    {
        "context": final_reranked_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# display_data()

context_precision_evaluation()

# query = "آئی ای ایل ٹی ایس (IELTS) امتحان کی تیاری کا بہترین طریقہ کیا ہے؟ "
# # query = "What is the best way to prepare for the IELTS exam?"
# fetched_docs = retrievar.invoke(query)
# print(f"Source Documents: {len(fetched_docs)}\n")
# for i, doc in enumerate(fetched_docs):
#     print(f"Document {i + 1}: {doc.page_content}\n")
# results = rag_chain.invoke(query)
# print("Answer: ", results)
