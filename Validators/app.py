import streamlit as st
import uvicorn
import transformers
import torch
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def load_model():
    with open('../models/valhalla-distilbart-mnli-12-6.pkl', 'rb') as file:
        valhalla_distilbart_mnli = joblib.load(file)
    return valhalla_distilbart_mnli


def main():
    st.title("Model Test Bed")
    st.sidebar.selectbox("Model", ("Model1", "Model2"))

    sequence_to_classify = st.text_area("Enter Sentence", "")
    candidate_labels = st.text_input("List of Biases (separated by ',')", "")
    if st.checkbox("Allow multiple correct topic"):
        multi_class = True
    else:
        multi_class = False
    valhalla_distilbart_mnli = load_model()
    result = valhalla_distilbart_mnli(sequence_to_classify, candidate_labels.strip('').split(','), multi_class=multi_class)
    st.json(result)

    # fig = px.bar(result, x='scores', y='labels', title='Confidence')
    # st.plotly_chart(fig)

    red_blue = ['#0A5F83','#0F7480','#28988A','#668B26', '#ABAB32']
    palette = sns.color_palette(red_blue)
    sns.set_palette(palette)
    sns.axes_style("white")
    sns.set(context='talk', style='dark', palette= red_blue , font_scale=1.3, color_codes=False, rc=None)
    f,ax = plt.subplots(figsize = (16,8))
    sns.barplot(y='labels', x='scores', data=result)
    sns.despine()
    st.pyplot(f)


if __name__ == '__main__':
    load_model()
    main()
