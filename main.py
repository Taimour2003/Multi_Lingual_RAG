from document_processing import document_processing
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# # # document_processing
# chunks_english = document_processing("./IELTS_Prepration.docx")
# chunks_urdu = document_processing("./IELTS_Preparation_Guide_Urdu.docx")
# chunks_arabic = document_processing("./IELTS_Preparation_Guide_Arabic.docx")

# total_chunks = chunks_english + chunks_urdu + chunks_arabic


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/BAAI/bge-m3")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# vector_store = Chroma.from_documents(
#     documents=total_chunks, embedding=embeddings, persist_directory="./vector_store2"
# )

vector_store = Chroma(
    persist_directory="./vector_store2", embedding_function=embeddings
)

retrievar = vector_store.as_retriever(search_kwargs={"k": 2})

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
    {"context": retrievar | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "آئی ای ایل ٹی ایس (IELTS) امتحان کی تیاری کا بہترین طریقہ کیا ہے؟ "
# query = "What is the best way to prepare for the IELTS exam?"
fetched_docs = retrievar.invoke(query)
for i, doc in enumerate(fetched_docs):
    print(f"Document {i + 1}: {doc.page_content}\n")
results = rag_chain.invoke(query)
print("Answer: ", results)
