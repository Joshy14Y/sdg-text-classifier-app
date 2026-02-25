import streamlit as st

from inference.model_loader import ModelLoader
from inference.text_preprocessor import TextPreprocessor


@st.cache_resource
def load_model():
    return ModelLoader()


@st.cache_resource
def load_preprocessor():
    return TextPreprocessor()


model = load_model()
preprocessor = load_preprocessor()

st.title("Clasificador de Objetivos de Desarrollo Sostenible")
st.write("Ingresa un texto en español para identificar el ODS al que pertenece.")

text_input = st.text_area("Texto")

if st.button("Clasificar"):
    if not text_input.strip():
        st.error("Por favor ingresa un texto.")
    else:
        processed = preprocessor.preprocess(text_input)
        prediction = model.predict(processed)
        st.success(prediction)
