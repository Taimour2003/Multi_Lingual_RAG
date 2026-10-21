from .clean_documents.clean_documents import cleaning_documents
from .loading_docs.loading_documents import load_document
from .split_documents.split_documents import spliting_document


def document_processing(file_path):
    docs = load_document(file_path)
    cleaned_docs = cleaning_documents(docs[0].page_content)
    # print("Number of Docs Available after cleaning = ", len(cleaned_docs))
    # print("Content of Cleaned Document = ", cleaned_docs)
    chunks = spliting_document(cleaned_docs)
    print("Content of Chunk 0 = ", chunks[0])
    print("Content of Chunk 1 = ", chunks[1])
    print("Content of Chunk 2 = ", chunks[2])
    print("Content of Chunk 3 = ", chunks[3])
    return chunks
