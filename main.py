from document_processing import document_processing
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# # document_processing
# chunks = document_processing("./IELTS_Prepration.docx")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector_store = Chroma.from_documents(
#     documents=chunks, embedding=embeddings, persist_directory="./vector_store"
# )

vector_store = Chroma(persist_directory="./vector_store", embedding_function=embeddings)

retrievar = vector_store.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1)


template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


rag_chain = (
    {"context": retrievar | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is the best way to prepare for IELTS exam?"
results = rag_chain.invoke(query)
print("Answer: ", results)
