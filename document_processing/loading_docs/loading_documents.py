from langchain_community.document_loaders import Docx2txtLoader


def load_document(file_path):
    print("Loading the document from the file path: ", file_path)
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    print("Number of Docs Available = ", len(docs))
    return docs
