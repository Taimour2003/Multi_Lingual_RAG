import asyncio

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevance,
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
)

# from ragas.metrics.collections import ContextRecall
from ragas.run_config import RunConfig
from test_data import invoke_dataset

data = invoke_dataset()
evaluation_dataset = Dataset.from_dict(data)


# def context_precision_evaluation():
#     # 1. Load data
#     # 2. Define local LLM with high timeout for local Ollama processing
#     local_llm = ChatOllama(
#         model="qwen2.5:1.5b",
#         temperature=0.0,
#         request_timeout=360.0,  # Gives Ollama enough time to generate responses
#     )
#     print("Local LLM initialized")

#     # 3. Wrap LLM for Ragas
#     wrapped_ragas_llm = LangchainLLMWrapper(local_llm)

#     # 4. Initialize Context Precision metric
#     context_precision_metric = LLMContextPrecisionWithReference(llm=wrapped_ragas_llm)
#     print("Context precision metric initialized")

#     my_run_config = RunConfig(max_workers=1, timeout=360)

#     # 5. Run evaluation sequentially (max_workers=1 prevents Ollama parallel crashes)
#     results = evaluate(
#         dataset=evaluation_dataset,
#         metrics=[context_precision_metric],
#         run_config=my_run_config,
#     )

#     print("Evaluation completed successfully!")

#     # 6. Convert to Pandas DataFrame
#     df = results.to_pandas()
#     print("Dataframe", df)

#     # # Correct column names based on new Ragas schema:
#     # metric_col = "llm_context_precision_with_reference"
#     # query_col = "user_input" if "user_input" in df.columns else "question"

#     # if metric_col in df.columns:
#     #     print(df[[query_col, metric_col]])
#     # else:
#     #     print(df)


# async def context_recall_evaluation():

#     client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
#     llm = llm_factory("qwen2.5:1.5b", client=client)

#     scorer = ContextRecall(llm=llm)

#     result = []

#     for data in evaluation_dataset:
#         temp_result = await scorer.ascore(
#             user_input=data["question"],
#             retrieved_contexts=data["contexts"],
#             reference=data["ground_truth"],
#         )
#         result.append(temp_result)

#     print("Context Recall Evaluation Result:", result)


async def run_full_rag_evaluation():
    # 1. Load dataset (Requires: user_input, retrieved_contexts, response, reference)
    data = invoke_dataset()
    evaluation_dataset = Dataset.from_dict(data)

    # 2. Setup Async client pointing to local Ollama server
    async_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Required placeholder for OpenAI-compatible client
    )

    # 3. Initialize local evaluator LLM using llm_factory
    evaluator_llm = llm_factory(model="qwen2.5:1.5b", client=async_client)

    # 4. Initialize all generation and retrieval metrics
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevance_metric = AnswerRelevance(llm=evaluator_llm)
    answer_correctness_metric = AnswerCorrectness(llm=evaluator_llm)
    precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
    recall_metric = ContextRecall(llm=evaluator_llm)

    # 5. Configure RunConfig for local execution (sequential processing + safe timeout)
    my_run_config = RunConfig(
        max_workers=1,  # Keep at 1 for local Ollama stability
        timeout=360,  # Extended timeout for generation and relevance checks
    )

    print("Starting full end-to-end RAG evaluation (Retrieval + Generation)...")

    # 6. Run evaluation across all metrics
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness_metric,
            answer_relevance_metric,
            answer_correctness_metric,
            precision_metric,
            recall_metric,
        ],
        run_config=my_run_config,
    )

    print("\nEvaluation completed successfully!")
    print(results)

    # 7. Convert results to a Pandas DataFrame for side-by-side analysis
    df = results.to_pandas()

    print("\nDetailed Per-Sample Breakdown:")
    print(
        df[
            [
                "user_input",
                "response",
                "faithfulness",
                "answer_relevance",
                "answer_correctness",
                "llm_context_precision_with_reference",
                "context_recall",
            ]
        ]
    )


if __name__ == "__main__":
    # context_precision_evaluation()
    # context_recall_evaluation()
    # asyncio.run(context_recall_evaluation())
    asyncio.run(run_full_rag_evaluation())
