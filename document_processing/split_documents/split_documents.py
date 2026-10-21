from langchain_text_splitters import RecursiveCharacterTextSplitter


def spliting_document(docs):
    # print("Number of Docs Available = ", len(docs))
    # print(f"Total character count: {len(docs[0])}")
    print(f"Complete Docs: {docs}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30, length_function=len
    )
    chunks = text_splitter.create_documents([docs])
    print("Number of Chunks Available = ", len(chunks))
    return chunks
