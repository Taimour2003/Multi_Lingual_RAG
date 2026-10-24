import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import umap
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def display_data():

    # 1. Load your vector store and embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = Chroma(
        persist_directory="./vector_store2", embedding_function=embeddings
    )

    # 2. Fetch all stored vectors and their metadata/content from Chroma
    data = vector_store.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(data["embeddings"])
    documents = data["documents"]

    # (Optional) Assume you have a way to tag languages, or if you want to inspect a specific set of concepts
    # For this demo, let's assume we extract vectors. Let's create dummy language labels if not stored in metadata:
    # In a real run, you can pull language tags from doc.metadata['lang'] or infer them.
    # Let's map them for demonstration:
    colors = []
    labels = []
    for doc in documents:
        # Simple heuristic or metadata check to color-code
        # If your metadata has language tags: doc['metadata'].get('lang')
        # Here we mock it based on script characters for illustration:
        if any("\u0600" <= c <= "\u06ff" for c in doc):  # Arabic/Urdu script block range
            # Check if it looks more like Urdu or Arabic, or group as 'Multilingual Script'
            colors.append("green")
            labels.append("Urdu/Arabic Chunks")
        else:
            colors.append("blue")
            labels.append("English Chunks")

    # 3. Reduce High-Dimensional Vectors (1024-d) down to 2D using UMAP
    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    embedding_2d = reducer.fit_transform(vectors)

    # 4. Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        alpha=0.7,
        s=50,
        edgecolors="k",
        linewidth=0.5,
    )

    plt.title(
        "UMAP Visualization of BGE-M3 Shared Multilingual Vector Space",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)

    # Custom legend

    eng_patch = mpatches.Patch(color="blue", label="English Chunks")
    ur_ar_patch = mpatches.Patch(color="green", label="Urdu/Arabic Chunks")
    plt.legend(handles=[eng_patch, ur_ar_patch], loc="upper right")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save or show
    plt.savefig("multilingual_vector_space.png", dpi=300)
    print("Plot saved as 'multilingual_vector_space.png'!")
    plt.show()