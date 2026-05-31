import streamlit as st

st.set_page_config(page_title="РГР: Классификация вин", layout="wide")
st.title(" Классификация типа вина по химическому составу")
st.header("Расчетно-графическая работа по дисциплине «Машинное обучение и большие данные»")

st.subheader("Автор")
col1, col2 = st.columns([1, 3])
with col1:
    st.image("my_photo.png", width=200)
with col2:
    st.write("**ФИО:** Бакланов Даниил Леонидович")
    st.write("**Группа:** МО-241")