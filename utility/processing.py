import stanza

stanza.download("uk")


def split_text(text: str, *, lowercase: bool = True, lemmatize: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()

    nlp = stanza.Pipeline("uk")
    doc = nlp(text.lower())

    tokens = (token for sentence in doc.sentences for token in sentence.words)
    words = (token for token in tokens if token.pos != "PUNCT")

    return [word.lemma if lemmatize else word.text for word in words]
