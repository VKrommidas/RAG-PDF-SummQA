
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Optional spaCy support
# --------------------------------------------------

try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
        USE_SPACY = True
        logger.info("spaCy English model loaded successfully.")

    except OSError:
        nlp = None
        USE_SPACY = False
        logger.warning(
            "spaCy is installed, but en_core_web_sm is unavailable. "
            "Using regex-based sentence segmentation."
        )

except ImportError:
    nlp = None
    USE_SPACY = False
    logger.info("spaCy is not installed. Using regex-based segmentation.")


# --------------------------------------------------
# Sentence processing
# --------------------------------------------------

def improve_sentence_continuity(
    text,
    redundant_phrases=None,
    transitions=None
):
    """
    Normalize sentence spacing and punctuation.

    Optionally removes explicitly supplied redundant phrases.
    Uses spaCy for sentence segmentation when available,
    otherwise falls back to regular expressions.
    """

    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Remove redundant phrases only when explicitly supplied.
    # Avoid automatically deleting phrases that may carry meaning.
    if redundant_phrases:
        for phrase in redundant_phrases:
            if phrase:
                text = re.sub(
                    re.escape(phrase),
                    "",
                    text,
                    flags=re.IGNORECASE
                )

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Sentence segmentation
    if USE_SPACY and nlp is not None:
        try:
            doc = nlp(text)
            sentences = [
                sentence.text.strip()
                for sentence in doc.sents
                if sentence.text.strip()
            ]

        except Exception as e:
            logger.warning(
                "spaCy sentence segmentation failed: %s. "
                "Using regex fallback.",
                e
            )
            sentences = re.split(r"(?<=[.!?])\s+", text)

    else:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Normalize whitespace within sentences.
    sentences = [
        re.sub(r"\s+", " ", sentence)
        for sentence in sentences
    ]

    # Join sentences without removing meaningful punctuation.
    result = " ".join(sentences)

    # Remove accidental spaces before punctuation.
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)

    # Fix repeated sentence-ending punctuation.
    result = re.sub(r"([.!?]){2,}", r"\1", result)

    result = re.sub(r"\s+", " ", result).strip()

    # Add final punctuation only when needed.
    if result and result[-1] not in ".!?":
        result += "."

    return result


# --------------------------------------------------
# PDF text cleaning
# --------------------------------------------------

def clean_text(text, lower=True):
    """
    Clean extracted PDF text while preserving scientific content.

    Normalizes Unicode, removes URLs and unwanted control
    characters, and normalizes whitespace.

    Set lower=False to preserve original capitalization.
    """

    if not text or not isinstance(text, str):
        return ""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs.
    text = re.sub(r"https?://\S+", "", text)

    # Remove unwanted control characters while preserving
    # normal whitespace.
    text = "".join(
        char for char in text
        if char.isprintable() or char in "\n\t\r"
    )

    # Normalize line breaks and whitespace.
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before common punctuation.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    text = text.strip()

    if lower:
        text = text.lower()

    return text


# --------------------------------------------------
# Raw PDF text extraction
# --------------------------------------------------

def extract_raw_text(pdf_path, clean=False):
    """
    Extract text from a PDF using PyPDF2.

    Optionally applies clean_text() to the extracted content.
    """

    try:
        import PyPDF2

    except ImportError as e:
        raise ImportError(
            "PyPDF2 is required for extract_raw_text. "
            "Install it using: pip install PyPDF2"
        ) from e

    text_parts = []

    with open(pdf_path, "rb") as file:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    text = "\n".join(text_parts)

    if clean:
        text = clean_text(text)

    return text.strip()


# --------------------------------------------------
# Unicode normalization
# --------------------------------------------------

def normalize_text(text, remove_non_ascii=False):
    """
    Normalize Unicode text.

    Optionally removes characters outside the ASCII range.
    """

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    if remove_non_ascii:
        text = text.encode(
            "ascii",
            "ignore"
        ).decode("ascii")

    return text


# --------------------------------------------------
# Token counting
# --------------------------------------------------

def token_count(text, tokenizer):
    """
    Return the number of tokens produced by a Hugging Face tokenizer.
    """

    if not text:
        return 0

    tokenized = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True
    )

    return len(tokenized["input_ids"])
