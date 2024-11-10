import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from utility.analysis import rank_words, zipf_fit
from utility.processing import split_text

# Інтерфейс Streamlit
st.title("Параметри закону Ципфа")

# Введення тексту користувачем
text = st.text_area("Введіть текст для отримання параметрів:",
                    "Цей текст є прикладом для аналізу закону Ципфа. У ньому повторюються слова для прикладу.")

# Виконуємо аналіз
if st.button("Запустити"):
    words = split_text(text)
    df = rank_words(words)
    words, ranks, frequencies = list(df["word"]), df["rank"], df["frequency"],
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
