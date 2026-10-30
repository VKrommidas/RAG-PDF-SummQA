
import logging
import re

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import clean_text, improve_sentence_continuity


logger = logging.getLogger(__name__)


class QueryHandler:

    def __init__(
        self,
        vector_store,
        summarization_pipeline,
        qa_pipeline,
        max_qa_tokens=256
    ):
        self.vector_store = vector_store
        self.summarization_pipeline = summarization_pipeline
        self.qa_pipeline = qa_pipeline
        self.max_qa_tokens = max_qa_tokens

    def handle_query(self, query_type, query, k=15):

        if query_type == "summarize":
            return self.summarize_pdf()

        elif query_type == "question":
            return self.answer_question(query, k)

        else:
            logger.error("Unknown query type: %s", query_type)
            return "Unknown query type."

    def get_dynamic_summary_lengths(
        self,
        text,
        default_min=50,
        default_max=300,
        multiplier=0.5
    ):
        """
        Determine suitable summary generation lengths based on
        the input token count.
        """

        tokenizer = self.summarization_pipeline.tokenizer

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            add_special_tokens=True
        )

        num_tokens = len(tokenized["input_ids"])

        if num_tokens == 0:
            return 1, 10

        if num_tokens < default_min:
            max_length = max(10, int(num_tokens * 0.8))
            min_length = max(5, min(15, max_length // 2))

        else:
            max_length = int(num_tokens * multiplier)
            max_length = min(
                max(max_length, default_min),
                default_max
            )

            min_length = min(default_min, max_length // 2)
            min_length = max(10, min_length)

        min_length = min(min_length, max_length - 1)

        return min_length, max_length

    def _summarize_text(
        self,
        text,
        default_min=50,
        default_max=300,
        multiplier=0.5
    ):
        """Summarize a single text input safely."""

        if not text or not text.strip():
            return ""

        min_length, max_length = self.get_dynamic_summary_lengths(
            text,
            default_min=default_min,
            default_max=default_max,
            multiplier=multiplier
        )

        try:
            result = self.summarization_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )

            if result and result[0].get("summary_text"):
                return result[0]["summary_text"].strip()

            return ""

        except Exception as e:
            logger.exception("Text summarization failed: %s", e)
            return ""

    def answer_question(self, query, k=15):

        if not query or not query.strip():
            return "Please provide a valid question."

        if self.vector_store.vector_store is None:
            logger.error("Vector store is not initialized.")

            return (
                "Vector store not available. "
                "Please ensure the PDFs are indexed properly."
            )

        # --------------------------------------------------
        # 1. Retrieve relevant documents
        # --------------------------------------------------

        try:
            retrieved_docs = self.vector_store.search(
                query,
                k=k
            )

        except Exception as e:
            logger.exception("Document retrieval failed: %s", e)
            return "An error occurred while retrieving relevant documents."

        if not retrieved_docs:
            return "No relevant documents were found."

        logger.info(
            "Retrieved %d relevant document chunks.",
            len(retrieved_docs)
        )

        # --------------------------------------------------
        # 2. Split retrieved documents into smaller chunks
        # --------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunk_score_pairs = []

        for doc in retrieved_docs:

            cleaned_text = clean_text(doc.page_content)

            if not cleaned_text:
                continue

            chunks = text_splitter.split_text(cleaned_text)

            for chunk in chunks:
                if chunk.strip():
                    chunk_score_pairs.append(chunk)

        if not chunk_score_pairs:
            return "No usable text was found in the retrieved documents."

        # --------------------------------------------------
        # 3. Build the context using retrieval order
        # --------------------------------------------------

        max_context_chars = 3000

        prioritized_chunks = []
        current_length = 0

        for chunk in chunk_score_pairs:

            remaining = max_context_chars - current_length

            if remaining <= 0:
                break

            selected_chunk = chunk[:remaining]

            prioritized_chunks.append(selected_chunk)
            current_length += len(selected_chunk)

        prioritized_context = " ".join(prioritized_chunks)

        logger.info(
            "Selected %d chunks from %d retrieved chunks.",
            len(prioritized_chunks),
            len(chunk_score_pairs)
        )

        logger.info(
            "Prioritized context length: %d characters.",
            len(prioritized_context)
        )

        if not prioritized_context.strip():
            return "No usable context was retrieved for this question."

        # --------------------------------------------------
        # 4. Compress the context using DistilBART
        # --------------------------------------------------

        final_context = self._summarize_text(
            prioritized_context,
            default_min=50,
            default_max=300,
            multiplier=0.5
        )

        # If summarization fails, use the original context.
        if not final_context:
            logger.warning(
                "Context summarization failed. Using original context."
            )
            final_context = prioritized_context

        final_context = improve_sentence_continuity(final_context)

        # --------------------------------------------------
        # 5. Construct the QA prompt
        # --------------------------------------------------

        prompt = f"""
You are a knowledgeable assistant.

Answer the question using only the information contained in the context.

Provide a clear and technically accurate explanation.
Use relevant facts, observations, and numerical values from the context.

Do not introduce external information or unsupported claims.

If the context does not contain enough information to answer
the question, clearly state that.

Question:
{query}

Context:
{final_context}

Answer:
"""

        # --------------------------------------------------
        # 6. Generate the answer using FLAN-T5
        # --------------------------------------------------

        try:
            qa_result = self.qa_pipeline(
                prompt,
                truncation=True,
                max_new_tokens=self.max_qa_tokens,
                do_sample=False
            )

            if (
                not qa_result
                or not isinstance(qa_result, list)
                or not qa_result[0].get("generated_text")
            ):
                logger.error("QA model returned an empty response.")
                return "No valid answer could be generated."

            answer = qa_result[0]["generated_text"].strip()

            answer = improve_sentence_continuity(answer)

            # Do not remove isolated numbers:
            # numerical values may be important to scientific answers.

            return answer if answer else "No meaningful answer was found."

        except Exception as e:
            logger.exception("Question answering failed: %s", e)
            return "An error occurred while generating the answer."

    def summarize_pdf(self):
        """
        Generate an analytical summary of all PDF documents
        indexed in the vector store.
        """

        # --------------------------------------------------
        # 1. Retrieve all indexed documents
        # --------------------------------------------------

        docs = self.vector_store.get_all_documents()

        if not docs:
            logger.warning("No documents are available for summarization.")
            return "No documents available to summarize."

        full_text = " ".join(
            doc.page_content
            for doc in docs
            if doc.page_content and doc.page_content.strip()
        )

        if not full_text.strip():
            return "No usable text was found in the indexed documents."

        logger.info(
            "Total document text length: %d characters.",
            len(full_text)
        )

        # --------------------------------------------------
        # 2. Split the full text into manageable chunks
        # --------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )

        split_texts = text_splitter.split_text(full_text)

        filtered_chunks = [
            chunk for chunk in split_texts
            if len(chunk.strip()) >= 100
        ]

        if not filtered_chunks:
            return "Insufficient text available for summarization."

        logger.info(
            "Generated %d text chunks for summarization.",
            len(filtered_chunks)
        )

        # --------------------------------------------------
        # 3. Summarize chunks in batches
        # --------------------------------------------------

        instruction = (
            "Please provide an analytical summary of the text below. "
            "Focus on wastewater treatment, key insights, critical data, "
            "and the most important details."
        )

        batch_size = 8
        summarized_chunks = []

        logger.info("Starting document summarization...")

        for i in tqdm(
            range(0, len(filtered_chunks), batch_size),
            desc="Summarizing documents"
        ):

            batch_chunks = filtered_chunks[i:i + batch_size]

            batch_inputs = [
                f"{instruction}\n\n{chunk}"
                for chunk in batch_chunks
            ]

            try:
                tokenizer = self.summarization_pipeline.tokenizer

                tokenized = tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )

                input_lengths = [
                    len(input_ids)
                    for input_ids in tokenized["input_ids"]
                ]

                batch_max_input = max(input_lengths)

                if batch_max_input < 20:
                    logger.info(
                        "Skipping batch %d because its input is too short.",
                        i
                    )
                    continue

                dynamic_max = min(
                    180,
                    max(30, int(batch_max_input * 0.4))
                )

                dynamic_min = min(
                    60,
                    max(10, dynamic_max // 3)
                )

                outputs = self.summarization_pipeline(
                    batch_inputs,
                    max_length=dynamic_max,
                    min_length=dynamic_min,
                    do_sample=False,
                    truncation=True
                )

                for output in outputs:

                    summary = output.get("summary_text", "").strip()

                    if summary:
                        summarized_chunks.append(summary)
                    else:
                        logger.warning(
                            "An empty summary was generated."
                        )

            except Exception as e:
                logger.exception(
                    "Error summarizing batch %d: %s",
                    i,
                    e
                )

        if not summarized_chunks:
            logger.error("No intermediate summaries were generated.")
            return "Summarization failed due to insufficient usable data."

        # --------------------------------------------------
        # 4. Combine intermediate summaries
        # --------------------------------------------------

        combined_summary = " ".join(summarized_chunks)

        # Remove the known Newsquiz artifact from the original workflow.
        combined_summary = re.sub(
            r"Use the weekly Newsquiz.*?Heroes\s*\**\.?",
            "",
            combined_summary,
            flags=re.IGNORECASE
        ).strip()

        if not combined_summary:
            return "No meaningful summary could be generated."

        logger.info(
            "Combined intermediate summary length: %d characters.",
            len(combined_summary)
        )

        # --------------------------------------------------
        # 5. Generate the final hierarchical summary
        # --------------------------------------------------

        final_instruction = (
            "Based on the following summary, provide a concise and "
            "analytical final summary that highlights the key insights, "
            "important findings, and critical details:"
        )

        hierarchical_input = (
            f"{final_instruction}\n\n{combined_summary}"
        )

        final_summary = self._summarize_text(
            hierarchical_input,
            default_min=100,
            default_max=350,
            multiplier=0.5
        )

        if not final_summary:
            logger.warning(
                "Final summarization failed. Returning intermediate summary."
            )
            return combined_summary

        return improve_sentence_continuity(final_summary)
