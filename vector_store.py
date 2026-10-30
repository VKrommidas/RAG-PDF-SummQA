
import os
import json
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import clean_text


logger = logging.getLogger(__name__)

FAISS_FILE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vectorstore"
)

MANIFEST_FILE = os.path.join(FAISS_FILE_DIRECTORY, "manifest.json")


class VectorStore:

    def __init__(self, persist=True):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )

        self.persist = persist
        self.vector_store = None

        if self.persist:
            os.makedirs(FAISS_FILE_DIRECTORY, exist_ok=True)

            index_file = os.path.join(
                FAISS_FILE_DIRECTORY, "index.faiss"
            )

            metadata_file = os.path.join(
                FAISS_FILE_DIRECTORY, "index.pkl"
            )

            if os.path.exists(index_file) and os.path.exists(metadata_file):
                try:
                    logger.info("Loading existing FAISS vector store...")

                    self.vector_store = FAISS.load_local(
                        FAISS_FILE_DIRECTORY,
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )

                    logger.info("FAISS vector store loaded successfully.")

                except Exception as e:
                    logger.exception(
                        "Could not load existing FAISS index: %s", e
                    )
                    self.vector_store = None

    def _get_pdf_manifest(self, pdf_paths):
        """Create a fingerprint of the PDF files used for indexing."""

        manifest = []

        for pdf_path in sorted(pdf_paths):

            absolute_path = os.path.abspath(pdf_path)

            try:
                file_stats = os.stat(absolute_path)

                manifest.append({
                    "path": absolute_path,
                    "size": file_stats.st_size,
                    "modified": file_stats.st_mtime_ns
                })

            except OSError as e:
                logger.warning(
                    "Could not access PDF %s: %s", absolute_path, e
                )

        return manifest

    def _load_manifest(self):
        """Load the metadata describing the previously indexed PDFs."""

        if not os.path.exists(MANIFEST_FILE):
            return None

        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
                return json.load(file)

        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read PDF manifest: %s", e)
            return None

    def _save_manifest(self, manifest):
        """Save the metadata describing the indexed PDFs."""

        os.makedirs(FAISS_FILE_DIRECTORY, exist_ok=True)

        temporary_file = MANIFEST_FILE + ".tmp"

        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4)

        os.replace(temporary_file, MANIFEST_FILE)

    def add_pdfs(self, pdf_paths, chunk_size=1000, chunk_overlap=200):
        """
        Index PDFs in the supplied folder.

        Reuse the existing index if the PDF files have not changed.
        Rebuild the index if PDFs have been added, removed, or modified.
        """

        pdf_paths = [
            os.path.abspath(path)
            for path in pdf_paths
            if os.path.isfile(path) and path.lower().endswith(".pdf")
        ]

        if not pdf_paths:
            logger.warning("No valid PDF files were provided.")
            self.vector_store = None
            return

        current_manifest = self._get_pdf_manifest(pdf_paths)
        saved_manifest = self._load_manifest() if self.persist else None

        # Reuse the existing index only when its source PDFs are unchanged.
        if (
            self.persist
            and self.vector_store is not None
            and saved_manifest == current_manifest
        ):
            logger.info(
                "PDF files have not changed. Reusing existing FAISS index."
            )
            return

        logger.info("Building FAISS index from PDF documents...")

        documents = []

        for pdf_path in pdf_paths:

            try:
                logger.info("Loading PDF: %s", pdf_path)

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                if not docs:
                    logger.warning(
                        "No text extracted from PDF: %s", pdf_path
                    )
                    continue

                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)

                # Exclude pages that contain no usable text.
                docs = [
                    doc for doc in docs
                    if doc.page_content.strip()
                ]

                documents.extend(docs)

            except Exception as e:
                logger.exception(
                    "Error processing PDF %s: %s", pdf_path, e
                )

        if not documents:
            logger.error("No usable PDF text was extracted.")
            self.vector_store = None
            return

        logger.info("Total extracted pages: %d", len(documents))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.error("No text chunks were generated.")
            self.vector_store = None
            return

        logger.info("Generated %d text chunks.", len(chunks))

        try:
            # Rebuild from all current PDFs to avoid stale or duplicate data.
            self.vector_store = FAISS.from_documents(
                chunks,
                self.embedding_model
            )

            logger.info("FAISS index successfully populated.")

            if self.persist:
                self._save_index()
                self._save_manifest(current_manifest)

        except Exception as e:
            logger.exception("Error creating FAISS index: %s", e)
            self.vector_store = None

    def _save_index(self):
        """Persist the FAISS index to the local project directory."""

        if self.vector_store is None:
            logger.warning("No FAISS index available to save.")
            return

        try:
            os.makedirs(FAISS_FILE_DIRECTORY, exist_ok=True)

            self.vector_store.save_local(FAISS_FILE_DIRECTORY)

            logger.info("FAISS index saved successfully.")

        except Exception as e:
            logger.exception("Error saving FAISS index: %s", e)

    def search(self, query, k=15):
        """
        Retrieve the k most relevant document chunks.

        FAISS returns L2 distances by default:
        lower distance means greater similarity.
        Results are returned in nearest-first order.
        """

        if self.vector_store is None:
            logger.warning("Vector store is not initialized.")
            return []

        try:
            results_with_scores = (
                self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
            )

            for i, (doc, distance) in enumerate(results_with_scores):
                logger.info(
                    "Retrieved chunk %d | L2 distance: %.4f",
                    i + 1,
                    float(distance)
                )

            # FAISS normally returns nearest-first results.
            # Sort explicitly to preserve that behavior.
            results_with_scores.sort(key=lambda item: item[1])

            return [doc for doc, distance in results_with_scores]

        except Exception as e:
            logger.exception(
                "Similarity search failed. Using basic search: %s", e
            )

            return self.vector_store.similarity_search(query, k=k)

    def get_all_documents(self):
        """Retrieve every document stored in the FAISS index."""

        if self.vector_store is None:
            return []

        try:
            document_ids = sorted(
                self.vector_store.index_to_docstore_id.items()
            )

            documents = []

            for index_position, document_id in document_ids:

                doc = self.vector_store.docstore.search(document_id)

                if not isinstance(doc, str):
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.exception(
                "Could not retrieve all indexed documents: %s", e
            )
            return []
