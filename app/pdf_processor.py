"""
pdf_processor.py

Handles PDF reading, text cleaning, and chunking.
"""

import re
from bidi.algorithm import get_display
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from flask_socketio import SocketIO  # only if you need to reference SocketIO
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A helper class to manage PDF reading, text cleaning, and chunking.
    """

    @staticmethod
    def clean_text(page_text: str) -> str:
        """
        Cleans the extracted text by:
          - Replacing multiple spaces with a single space
          - Removing hyphenated line breaks
          - Trimming whitespace
        """
        cleaned = re.sub(r"\s+", " ", page_text)
        cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
        return cleaned.strip()

    @staticmethod
    def chunk_text(text: str, max_words: int = 300, overlap_sentences: int = 1) -> list:
        """
        Splits the input text into chunks of sentences without exceeding max_words per chunk.
        Overlaps between chunks are handled by repeating the last `overlap_sentences` sentences
        of the previous chunk.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        overlap = []

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            # If adding the sentence exceeds max_words limit, start a new chunk
            if current_word_count + sentence_word_count > max_words:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_chunk = overlap.copy()
                    current_word_count = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def is_sentence_complete(text: str) -> bool:
        """
        Checks if the text ends with . ! or ? to determine completeness.
        """
        text = text.strip()
        return bool(text) and text[-1] in {'.', '!', '?'}

    @staticmethod
    def is_uppercase(text: str, threshold: float = 0.8) -> bool:
        """
        Determines if a significant portion of the text is uppercase.
        """
        if not text:
            return False
        uppercase_chars = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        if total_letters == 0:
            return False
        return (uppercase_chars / total_letters) >= threshold

    @staticmethod
    def pretty_print_filename(filename: str) -> str:
        """
        Converts a filename into a citation-friendly camel case format.
        """
        import os
        base, _ = os.path.splitext(filename)
        import re
        words = re.split(r'[_\- ]+', base.strip())
        return ''.join(word.capitalize() for word in words)

    @classmethod
    def chunk_pdf_text(
        cls,
        pdf_file_path: str,
        original_file_name: str,
        socketio_instance: SocketIO,
        socket_id: str,
        max_words: int = 300,
        overlap_sentences: int = 1,
        min_sentences_per_page: int = 3,
        uppercase_threshold: float = 0.8
    ) -> list:
        """
        Reads a PDF, processes text page-by-page, and returns a list of chunk dictionaries.
        """
        pdf_chunks = []
        residual_fragment = ""

        with pdfplumber.open(pdf_file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_number, page in enumerate(pdf.pages):
                # Emit progress every 10 pages or the last page
                if (page_number + 1) % 10 == 0 or (page_number + 1) == total_pages:
                    progress = (page_number + 1) / total_pages * 90
                    socketio_instance.emit('processing_progress', {'progress': progress}, room=socket_id)
                    socketio_instance.emit(
                        'status',
                        {'message': f'Chunking page {page_number + 1} of {total_pages}...'},
                        room=socket_id
                    )

                text = page.extract_text()
                if text:
                    # Convert left to right languages texts
                    lines = text.split("\n")
                    lines = [get_display(line) for line in lines]
                    text = "\n".join(lines)

                    # Append any leftover from the previous page
                    if residual_fragment:
                        text = residual_fragment + " " + text
                        residual_fragment = ""

                    cleaned_text = cls.clean_text(text)
                    sentences = sent_tokenize(cleaned_text)
                    sentence_count = len(sentences)

                    # Check if page is predominantly uppercase
                    upper_check = cls.is_uppercase(cleaned_text, threshold=uppercase_threshold)

                    # Decide if it's a 'regular' page or a header/title page
                    is_regular_page = (sentence_count >= min_sentences_per_page) and (not upper_check)

                    if is_regular_page:
                        # If the last sentence is incomplete, keep it for the next round
                        if not cls.is_sentence_complete(cleaned_text) and sentences:
                            residual_fragment = sentences.pop(-1)
                            cleaned_text = ' '.join(sentences)
                    else:
                        # Possibly a header or title page; clear any leftovers
                        residual_fragment = ""

                    # Chunk the text
                    chunks = cls.chunk_text(cleaned_text, max_words, overlap_sentences)
                    for chunk_id, chunk in enumerate(chunks):
                        pdf_chunks.append({
                            "pdf_name": cls.pretty_print_filename(original_file_name),
                            "page": page_number + 1,
                            "chunk_id": chunk_id,
                            "text": chunk
                        })

            # Handle any leftover sentence after processing all pages
            if residual_fragment:
                chunks = cls.chunk_text(residual_fragment, max_words, overlap_sentences)
                for chunk_id, chunk in enumerate(chunks):
                    pdf_chunks.append({
                        "pdf_name": cls.pretty_print_filename(original_file_name),
                        "page": total_pages,
                        "chunk_id": chunk_id,
                        "text": chunk
                    })

        return pdf_chunks
