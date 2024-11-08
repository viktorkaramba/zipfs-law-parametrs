import re
import streamlit as st
from collections import Counter
import numpy as np
from scipy.optimize import curve_fit
from streamlit_echarts import st_echarts
import pandas as pd


def preprocess_text(text):
    # Видаляємо спеціальні символи та переводимо текст у нижній регістр
    text = re.sub(r'\W+', ' ', text)
    return text.lower()


def zipf_law_analysis(text):
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


def zipf_fit(ranks, frequencies):
    # Функція для апроксимації закону Ципфа
    def zipf_func(rank, a):
        return a / rank

    # Апроксимація параметра 'a'
    params, _ = curve_fit(zipf_func, ranks, frequencies)
    return params[0]


# Інтерфейс Streamlit
st.title("Параметри закону Ципфа")

# Введення тексту користувачем
text = st.text_area("Введіть текст для отримання параметрів:",
                    "Цей текст є прикладом для аналізу закону Ципфа. У ньому повторюються слова для прикладу.")

# Виконуємо аналіз
if st.button("Запустити"):
    ranks, frequencies, words = zipf_law_analysis(text)
    a = zipf_fit(ranks, frequencies)

    # Підготовка даних для графіка
    fitted_frequencies = [a / rank for rank in ranks]

    # Побудова графіка у вигляді лінії через середини площадок емпіричних даних
    option = {
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": words[:20],  # Показуємо топ-20 слів за частотою
            "name": "Ранг",
        },
        "yAxis": {
            "type": "value",
            "name": "Частота"
        },
        "series": [
            {
                "data": frequencies[:20].tolist(),
                "type": "line",
                "name": "Емпіричні дані",
                "lineStyle": {"color": "blue"},
                "markPoint": {
                    "data": [
                        {"type": "average", "name": "Середня частота"}
                    ]
                },
            },
            {
                "data": fitted_frequencies[:20],
                "type": "line",
                "name": "Пряма Ципфа",
                "lineStyle": {"type": "dashed", "color": "red"},
                "smooth": True,
            },
        ],
    }

    st.write(f"Значення параметра a: {a}")
    st_echarts(options=option, height="400px")

    # Створення таблиці з даними
    data = {
        "Ранг": ranks[:20],
        "Слово": words[:20],
        "Частота": frequencies[:20],
        "Теоретична частота (Ципф)": np.round(fitted_frequencies[:20], 2)
    }
    df = pd.DataFrame(data)

    st.write("Таблиця частотного розподілу:")
    st.table(df)
