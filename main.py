
import os
import glob
import logging

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from vector_store import VectorStore
from query_handler import QueryHandler


logger = logging.getLogger(__name__)


# Hugging Face models
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
QA_MODEL = "google/flan-t5-large"


def load_models():
    """
    Load the summarization and question-answering models.

    Automatically uses CUDA when available and CPU otherwise.
    """

    device = 0 if torch.cuda.is_available() else -1

    if device == 0:
        print("Using GPU:", torch.cuda.get_device_name(0))
        model_dtype = torch.float16
    else:
        print("Using CPU.")
        model_dtype = torch.float32

    logger.info("Loading summarization model: %s", SUMMARIZATION_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        SUMMARIZATION_MODEL
    )

    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARIZATION_MODEL,
        torch_dtype=model_dtype
    )

    summarization_pipeline = pipeline(
        "summarization",
        model=summarization_model,
        tokenizer=tokenizer,
        device=device
    )

    logger.info("Loading question-answering model: %s", QA_MODEL)

    qa_pipeline = pipeline(
        "text2text-generation",
        model=QA_MODEL,
        device=device
    )

    logger.info("Both models loaded successfully.")

    return summarization_pipeline, qa_pipeline


def main():

    print("\nWelcome to the AI-Powered PDF QA System")
    print("=" * 45)

    # --------------------------------------------------
    # 1. Locate PDF documents
    # --------------------------------------------------

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(base_dir, "pdfs")

    os.makedirs(pdf_folder, exist_ok=True)

    pdf_paths = sorted(
        glob.glob(os.path.join(pdf_folder, "*.pdf"))
    )

    if not pdf_paths:
        print(
            "\nNo PDF files found in the 'pdfs' folder."
        )
        print("Please add at least one PDF and run the application again.")
        return

    print("\nFound PDF documents:")

    for pdf_path in pdf_paths:
        print("-", os.path.basename(pdf_path))

    # --------------------------------------------------
    # 2. Initialize and populate the vector store
    # --------------------------------------------------

    print("\nInitializing vector store...")

    try:
        vector_store = VectorStore()
        vector_store.add_pdfs(pdf_paths)

    except Exception as e:
        logger.exception("Vector store initialization failed: %s", e)
        print("\nError: Could not initialize the vector store.")
        return

    if vector_store.vector_store is None:
        print(
            "\nError: No searchable documents were indexed. "
            "Check the PDF files and application logs."
        )
        return

    print("PDF documents indexed successfully.")

    # --------------------------------------------------
    # 3. Load language models
    # --------------------------------------------------

    print("\nLoading language models. This may take some time...")

    try:
        summarization_pipeline, qa_pipeline = load_models()

    except Exception as e:
        logger.exception("Model loading failed: %s", e)
        print(
            "\nError: Could not load the language models. "
            "Check your internet connection, dependencies, "
            "and available memory."
        )
        return

    # --------------------------------------------------
    # 4. Initialize query handler
    # --------------------------------------------------

    query_handler = QueryHandler(
        vector_store,
        summarization_pipeline,
        qa_pipeline
    )

    print("\nSystem ready.")
    print("=" * 45)

    # --------------------------------------------------
    # 5. Interactive command-line interface
    # --------------------------------------------------

    while True:

        try:
            query_type = input(
                "\nEnter query type (summarize/question/exit): "
            ).strip().lower()

            if query_type == "exit":
                print("\nExiting the system. Goodbye!")
                break

            if query_type not in {"summarize", "question"}:
                print(
                    "Invalid query type. "
                    "Please enter 'summarize', 'question', or 'exit'."
                )
                continue

            if query_type == "summarize":
                query = "Summarize the indexed PDF documents."

            else:
                query = input("\nEnter your question: ").strip()

                if not query:
                    print("Error: Question cannot be empty.")
                    continue

            print("\nProcessing your request...")

            result = query_handler.handle_query(
                query_type,
                query
            )

            if query_type == "summarize":
                print("\nSummary:\n")
            else:
                print("\nAnswer:\n")

            print(result)

        except KeyboardInterrupt:
            print("\n\nApplication interrupted. Goodbye!")
            break

        except Exception as e:
            logger.exception("Error processing request: %s", e)
            print("\nAn error occurred while processing your request.")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    main()
