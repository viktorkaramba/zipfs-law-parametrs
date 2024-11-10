from collections import Counter

import numpy as np

from utility.text_processing import preprocess_text


def zipf_law_analysis(text: str) -> tuple[np.ndarray[int], np.ndarray[int], list[str]]:
    # Попередня обробка тексту
    processed_text = preprocess_text(text)

    # Розбиваємо текст на слова
    words = processed_text.split()

    # Рахуємо частоти кожного слова
    word_counts = Counter(words)

    # Сортуємо слова за частотою у порядку спадання
    sorted_word_counts = word_counts.most_common()

    # Витягуємо ранги і частоти
    ranks = np.arange(1, len(sorted_word_counts) + 1)
    frequencies = np.array([count for _, count in sorted_word_counts])

    return ranks, frequencies, [word for word, _ in sorted_word_counts]
