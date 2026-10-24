from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

dataset = [
    {
        "id": "EN01",
        "language": "en",
        "question": "How long do the four IELTS test sections take?",
        "expected_source_docs": ["IELTS_EN_S02"],
        "expected_facts": [
            "Listening approximately 30 minutes",
            "Paper-based answer transfer adds 10 minutes",
            "Reading 60 minutes",
            "Writing 60 minutes",
            "Speaking approximately 11-14 minutes",
        ],
        "type": "same_language",
    },
    {
        "id": "EN02",
        "language": "en",
        "question": "What is the difference between IELTS Academic and General Training?",
        "expected_source_docs": ["IELTS_EN_S01"],
        "expected_facts": [
            "Academic is commonly used for higher education",
            "General Training is commonly used for migration and work",
            "Requirements depend on the institution or authority",
        ],
        "type": "same_language",
    },
    {
        "id": "UR01",
        "language": "ur",
        "question": "IELTS کے چاروں حصوں کو مکمل کرنے میں کتنا وقت لگتا ہے؟",
        "expected_source_docs": ["IELTS_UR_S02"],
        "expected_facts": [
            "Listening approximately 30 minutes",
            "Reading 60 minutes",
            "Writing 60 minutes",
            "Speaking approximately 11-14 minutes",
        ],
        "type": "same_language",
    },
    {
        "id": "UR02",
        "language": "ur",
        "question": "IELTS Academic اور General Training میں کیا فرق ہے؟",
        "expected_source_docs": ["IELTS_UR_S01"],
        "expected_facts": [
            "Academic commonly used for higher education",
            "General Training commonly used for migration and work",
            "Choose according to official requirements",
        ],
        "type": "same_language",
    },
    {
        "id": "AR01",
        "language": "ar",
        "question": "كم يستغرق كل قسم من أقسام اختبار IELTS؟",
        "expected_source_docs": ["IELTS_AR_S02"],
        "expected_facts": [
            "Listening approximately 30 minutes",
            "Reading 60 minutes",
            "Writing 60 minutes",
            "Speaking approximately 11-14 minutes",
        ],
        "type": "same_language",
    },
    {
        "id": "AR02",
        "language": "ar",
        "question": "ما الفرق بين IELTS Academic وIELTS General Training؟",
        "expected_source_docs": ["IELTS_AR_S01"],
        "expected_facts": [
            "Academic commonly used for higher education",
            "General Training commonly used for migration and work",
            "Choose according to official requirements",
        ],
        "type": "same_language",
    },
    {
        "id": "XR01",
        "language": "ur",
        "question": "IELTS Writing Task 2 کو کتنا وزن دیا جاتا ہے؟",
        "expected_source_docs": ["IELTS_EN_S06", "IELTS_UR_S06"],
        "expected_facts": [
            "Task 2 contributes twice as much as Task 1 to the Writing score"
        ],
        "type": "cross_language",
    },
    {
        "id": "XR02",
        "language": "ar",
        "question": "كم تبلغ مدة اختبار القراءة؟",
        "expected_source_docs": ["IELTS_EN_S02", "IELTS_AR_S02"],
        "expected_facts": ["Reading lasts 60 minutes"],
        "type": "cross_language",
    },
    {
        "id": "XR03",
        "language": "en",
        "question": "What should I practise to improve IELTS speaking fluency?",
        "expected_source_docs": ["IELTS_UR_S07", "IELTS_AR_S07"],
        "expected_facts": [
            "Fluency and coherence",
            "Vocabulary",
            "Grammar",
            "Pronunciation",
        ],
        "type": "cross_language",
    },
]

processed_samples = []
ragas_dataset_dict = {"question": [], "contexts": [], "ground_truth": []}

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/BAAI/bge-m3")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store = Chroma(
    persist_directory="./vector_store2", embedding_function=embeddings
)
retrievar = vector_store.as_retriever(search_kwargs={"k": 5})


def invoke_dataset():
    for item in dataset:
        question = item["question"]
        # print("Question: ", question)
        # retrieved_contexts = retrievar.invoke(question)
        # print("Retrieved Contexts: ", retrieved_contexts)
        retrieved_docs = vector_store.similarity_search(question, k=3)
        # Combine expected facts into a single ground truth reference string
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        ground_truth_reference = " ".join(item["expected_facts"])
        # print("Ground Truth Reference: ", ground_truth_reference)
        # Append to dictionary structure
        ragas_dataset_dict["question"].append(question)
        ragas_dataset_dict["contexts"].append(retrieved_contexts)
        ragas_dataset_dict["ground_truth"].append(ground_truth_reference)

        # print(ragas_dataset_dict)

    return ragas_dataset_dict
