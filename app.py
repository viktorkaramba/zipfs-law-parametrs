import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from utility.analysis import rank_words, fit_zipf
from utility.factory import create_zipf_vectorized
from utility.processing import split_text


def build_zipf_chart(ranks: np.ndarray[float], frequencies: np.ndarray[float],
                     fitted_frequencies: np.ndarray[float]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.log(ranks),
        y=np.log(frequencies),
        mode="markers",
        name="Частота (емпірична)"
    ))

    fig.add_trace(go.Scatter(
        x=np.log(ranks),
        y=np.log(fitted_frequencies),
        mode="lines",
        name="Частота (модельна)"
    ))

    fig.update_layout(
        title="Log-Log графік закону Ципфа",
        xaxis={"title": "ln(Ранг)", "tickvals": np.log(ranks), "ticktext": [f"ln({r})" for r in ranks]},
        yaxis={"title": "ln(Частота)", "tickvals": np.log(frequencies), "ticktext": [f"ln({f})" for f in frequencies]},
        showlegend=True
    )

    return fig


# Інтерфейс Streamlit
st.title("Параметри закону Ципфа")

# Введення тексту користувачем
text = st.text_area("Введіть текст для отримання параметрів:",
                    "Цей текст є прикладом для аналізу закону Ципфа. У ньому повторюються слова для прикладу.")

# Виконуємо аналіз
if st.button("Запустити"):
    words = split_text(text)
    ranked_words_df = rank_words(words)
    ranks_df = ranked_words_df.drop("word", axis="columns").drop_duplicates()

    s, c = fit_zipf(ranks_df["rank"], ranks_df["frequency"])
    zipf_vectorized = create_zipf_vectorized(s, c)

    st.latex(r"f=\frac{C}{r^s}")
    st.latex(f"s={s},C={c}")

    zipf_chart = build_zipf_chart(ranks_df["rank"], ranks_df["frequency"], zipf_vectorized(ranks_df["rank"]))
    st.plotly_chart(zipf_chart)

    # Створення таблиці з даними
    st.write("Таблиця частотного розподілу:")
    st.table(
        pd.DataFrame({
            "Слово": ranked_words_df["word"],
            "Ранг": ranked_words_df["rank"],
            "Частота": ranked_words_df["frequency"],
            "Теоретична частота (Ципф)": np.round(zipf_vectorized(ranked_words_df["rank"]), 2)
        })
    )
