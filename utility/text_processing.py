import re


def preprocess_text(text: str) -> str:
    # Видаляємо спеціальні символи та переводимо текст у нижній регістр
    text = re.sub(r'\W+', ' ', text)
    return text.lower()
