import re

import spacy
from unidecode import unidecode


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_lg", exclude=["ner", "parser"])

    def _clean(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z\s]", "", text)

    def preprocess(self, text: str) -> str:
        text = text.lower().strip()
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not (token.is_stop or token.is_punct or token.is_space):
                clean_lemma = self._clean(unidecode(token.lemma_))
                if clean_lemma.strip():
                    tokens.append(clean_lemma)
        return " ".join(tokens)


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    test_text = "El acceso al agua potable es un derecho fundamental para el desarrollo sostenible."
    result = preprocessor.preprocess(test_text)
    print(f"Input: {test_text}")
    print(f"Output: {result}")
