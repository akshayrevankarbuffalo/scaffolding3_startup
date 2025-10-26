"""
starter_preprocess.py
Starter code for text preprocessing - focus on the statistics, not the regex!

This is the same code you'll use in the main Shannon assignment next week.
(Alternative Implementation)
"""

import re
import json
import requests
from typing import List, Dict, Tuple
from collections import Counter
import string


class TextPreprocessor:
    """Handles all the annoying text cleaning so you can focus on the fun stuff"""

    def __init__(self):
        # Gutenberg markers (these are common, add more if needed)
        self.gutenberg_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
            "<<THIS ELECTRONIC VERSION"
        ]

    def clean_gutenberg_text(self, raw_text: str) -> str:
        """Remove Project Gutenberg headers/footers"""
        lines = raw_text.split('\n')

        # Find start and end markers
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if any(marker in line for marker in self.gutenberg_markers[:4]):
                if "START" in line:
                    start_idx = i + 1
                elif "END" in line:
                    end_idx = i
                    break

        # Join the cleaned lines
        cleaned = '\n'.join(lines[start_idx:end_idx])

        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        return cleaned.strip()

    def normalize_text(self, text: str, preserve_sentences: bool = True) -> str:
        """
        Normalize text while preserving sentence boundaries

        Args:
            text: Input text
            preserve_sentences: If True, keeps . ! ? for sentence detection
        """
        # Convert to lowercase
        text = text.lower()

        # Standardize quotes and dashes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['’]", "'", text)
        text = re.sub(r'—|–', '-', text)

        if preserve_sentences:
            # Keep sentence endings but remove other punctuation
            # This regex keeps . ! ? but removes , ; : etc
            text = re.sub(r'[^\w\s.!?\'-]', ' ', text)
        else:
            # Remove all punctuation except apostrophes in contractions
            text = re.sub(r"(?<!\w)'(?!\w)|[^\w\s]", ' ', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (you can make this fancier with NLTK)
        # NOTE: This is the original one from the starter. Your completed
        # code uses a better regex, which is what I'll use in stats/summary.
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words"""
        # Remove sentence endings for word tokenization
        text_for_words = re.sub(r'[.!?]', '', text)

        # Split on whitespace and filter empty strings
        words = text_for_words.split()
        words = [w for w in words if w]

        return words

    def tokenize_chars(self, text: str, include_space: bool = True) -> List[str]:
        """Split text into characters"""
        if include_space:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            return list(text)
        else:
            return [c for c in text if c != ' ']

    def get_sentence_lengths(self, sentences: List[str]) -> List[int]:
        """Get word count for each sentence"""
        return [len(self.tokenize_words(sent)) for sent in sentences]

    # === ALTERNATIVE IMPLEMENTATIONS FOR TODO METHODS ===

    def fetch_from_url(self, url: str) -> str:
        """
        Fetch text content from a URL (especially Project Gutenberg)

        Args:
            url: URL to a .txt file

        Returns:
            Raw text content

        Raises:
            Exception if URL is invalid or cannot be reached
        """
        # Validate that it's a .txt URL
        if not isinstance(url, str) or not url.strip().lower().endswith(".txt"):
            raise ValueError("Invalid URL provided. Must be a .txt file link.")

        try:
            # Fetch content with a timeout
            page_request = requests.get(url, timeout=20)
            # Raise an exception for bad status codes (like 404, 500)
            page_request.raise_for_status()
        except requests.exceptions.RequestException as err:
            # Catch network/HTTP errors
            raise Exception(f"Failed to retrieve content from URL: {err}")
        except Exception as e:
            # Catch other potential errors
            raise Exception(f"An unexpected error occurred during fetch: {e}")

        return page_request.text

    def get_text_statistics(self, text: str) -> Dict:
        """
        Calculate basic statistics about the text

        Returns dictionary with:
            - total_characters
            - total_words  
            - total_sentences
            - avg_word_length
            - avg_sentence_length
            - most_common_words (top 10)
        """

        # Tokenize words: find all alphanumeric sequences
        word_list = re.findall(r"\b\w+\b", text.lower())

        # Tokenize sentences: split after punctuation, keeping it
        sentence_list = [s for s in re.split(
            r"(?<=[.!?])\s+", text.strip()) if s]

        # Get counts
        count_chars = len(text)
        count_words = len(word_list)
        count_sentences = len(sentence_list)

        # Calculate averages, handling div by zero with max(1, ...)
        sum_of_word_lengths = sum(len(word) for word in word_list)
        average_word_length = round(
            sum_of_word_lengths / max(1, count_words), 2)
        average_sentence_length = round(
            count_words / max(1, count_sentences), 2)

        # Get most common words
        word_frequencies = Counter(word_list)
        top_10_common = [word for word,
                         count in word_frequencies.most_common(10)]

        # Compile final dictionary
        stats_results = {
            "total_characters": count_chars,
            "total_words": count_words,
            "total_sentences": count_sentences,
            "avg_word_length": average_word_length,
            "avg_sentence_length": average_sentence_length,
            "most_common_words": top_10_common
        }
        return stats_results

    def create_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Create a simple extractive summary by returning the first N sentences

        Args:
            text: Cleaned text
            num_sentences: Number of sentences to include

        Returns:
            Summary string
        """
        # Tokenize into sentences using the more robust regex
        sentence_list = [s for s in re.split(
            r"(?<=[.!?])\s+", text.strip()) if s]

        # Select the first 'num_sentences'
        summary_list = sentence_list[:num_sentences]

        # Rejoin with spaces
        return " ".join(summary_list)


class FrequencyAnalyzer:
    """Calculate n-gram frequencies from tokenized text"""

    def calculate_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """
        Calculate n-gram frequencies

        Args:
            tokens: List of tokens (words or characters)
            n: Size of n-gram (1=unigram, 2=bigram, 3=trigram)

        Returns:
            Dictionary mapping n-grams to their counts
        """
        if n == 1:
            # Special case for unigrams (return as single strings, not tuples)
            return dict(Counter(tokens))

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)

        return dict(Counter(ngrams))

    def calculate_probabilities(self, ngram_counts: Dict, smoothing: float = 0.0) -> Dict:
        """
        Convert counts to probabilities

        Args:
            ngram_counts: Dictionary of n-gram counts
            smoothing: Laplace smoothing parameter (0 = no smoothing)
        """
        total = sum(ngram_counts.values()) + smoothing * len(ngram_counts)

        probabilities = {}
        for ngram, count in ngram_counts.items():
            probabilities[ngram] = (count + smoothing) / total

        return probabilities

    def save_frequencies(self, frequencies: Dict, filename: str):
        """Save frequency dictionary to JSON file"""
        # Convert tuples to strings for JSON serialization
        json_friendly = {}
        for key, value in frequencies.items():
            if isinstance(key, tuple):
                json_friendly['||'.join(key)] = value
            else:
                json_friendly[key] = value

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_friendly, f, indent=2, ensure_ascii=False)

    def load_frequencies(self, filename: str) -> Dict:
        """Load frequency dictionary from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Convert string keys back to tuples where needed
        frequencies = {}
        for key, value in json_data.items():
            if '||' in key:
                frequencies[tuple(key.split('||'))] = value
            else:
                frequencies[key] = value

        return frequencies


# Example usage to test your setup
if __name__ == "__main__":
    # Test with a small example
    example_text = """
    This is a test. This is only a test! 
    If this were a real emergency, you would be informed.
    """

    proc = TextPreprocessor()
    analyst = FrequencyAnalyzer()

    # Clean and normalize
    normalized = proc.normalize_text(example_text)
    print(f"Normalized text: {normalized}\n")

    # Test new methods
    statistics = proc.get_text_statistics(normalized)
    print(f"Statistics: {statistics}\n")

    summary_text = proc.create_summary(normalized, num_sentences=2)
    print(f"Summary: {summary_text}\n")

    # Test old methods
    words_list = proc.tokenize_words(normalized)
    print(f"Words: {words_list}\n")

    bigrams_dict = analyst.calculate_ngrams(words_list, 2)
    print(f"Word bigrams (first 5): {list(bigrams_dict.items())[:5]}")

    print("\n✅ All methods tested and working!")
