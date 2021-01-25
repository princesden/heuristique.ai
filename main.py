import streamlit as st
from annotated_text import annotated_text
import nltk
import joblib
import random
import time

# **************** PAGE CONFIG *************
st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="Heuristique",  # String or None. Strings get appended with "• Streamlit".
    page_icon="logo.png",  # String, anything supported by st.image, or None.
)


# *********************UTILITY*********************

def load_model():
    with open('models/valhalla-distilbart-mnli-12-6.pkl', 'rb') as file:
        model = joblib.load(file)
    return model


def add_colors():
    number_of_colors = len(bias_selection)
    color_hex = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    index = 0
    for bias in bias_selection:
        colors[bias] = color_hex[index]
        index += 1


def add_colors_set():
    color_list = ["#8ef", "#faa", "#fea", "#afa", "8dc"]
    index = 0
    for bias in bias_selection:
        colors[bias] = color_list[index]
        index += 1


def configure_individual_bias():
    is_configure_individual_bias = st.sidebar.checkbox('Configure Biases Individually')
    return is_configure_individual_bias


def pick_classification_color():
    color_pick = st.sidebar.color_picker('Pick output color for each Bias')
    return color_pick


def bias_wiki():
    st.sidebar.radio('Wiki', ['Definitions'])


# ******************MAIN WINDOW*********************

def parse_input_text():
    if analysis_boundary == 'Sentence':
        sentences = [x.replace('\n', '') for x in nltk.sent_tokenize(input_text)]
        return sentences
    else:
        corpus = input_text
        return corpus


def execute_ml():
    prediction_set = []
    model = load_model()
    tokens_to_classify = parse_input_text()
    for token in tokens_to_classify:
        prediction = model(token, bias_selection, multi_class=True)
        prediction_set.append(prediction)
    return prediction_set


def construct_annotation_elements(prediction):
    sequence = prediction['sequence']
    labels = prediction['labels'][0]
    scores = prediction['scores'][0]

    if (scores * 100) >= output_sensitivity:
        return sequence, labels, colors.get(labels)
    else:
        return sequence


def progress_bar(increase_by):
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(increase_by)


def write_output():
    add_colors_set()
    result = []
    prediction_set = execute_ml()
    increase_by = 0
    for prediction in prediction_set:
        # st.json(prediction)
        result.append(construct_annotation_elements(prediction))
        progress_bar(increase_by)
        increase_by += round(100/len(prediction_set))
    st.subheader('Result')
    annotated_text(*result, height=1000)
    progress_bar(100)


if __name__ == '__main__':

    # ******************MAIN BOARD*********************

    st.title('Find Your Cognitive Minefields')
    st.write('A **cognitive bias** is a systematic pattern of deviation from **norm** or **rationality** in judgment. '
             'Individuals create their own "subjective reality" from their perception of the input. An individual\'s '
             'construction of reality, not the objective input, may dictate their behavior in the world. Thus, '
             'cognitive biases may sometimes lead to perceptual distortion, inaccurate judgment, illogical '
             'interpretation, or what is broadly called irrationality. '
             )
    input_text = st.text_area('Enter Text', height=200, max_chars=10000)

    # ****************SIDEBAR / CONFIGURATIONS*********

    st.sidebar.image('logo.png', width=300)
    bias_selection = st.sidebar.multiselect('Select biases',
                                            ["Dunning-Kruger effect", "Bandwagon effect", "Negative bias",
                                             "Illusory correlation", "Overconfidence effect"])
    if len(bias_selection) == 0:
        bias_selection = ["Dunning-Kruger effect", "Bandwagon effect", "Negative bias", "Illusory correlation",
                          "Overconfidence effect"]

    output_sensitivity = st.sidebar.select_slider('Algorithm Sensitivity', options=[50, 60, 70, 80, 90, 100])
    is_multiple_correct_answers = st.sidebar.checkbox('Multiple correct answers')
    analysis_boundary = st.sidebar.radio('Analysis Boundary', ["Sentence", "Corpus"])
    colors = {}

    # ************************** EXECUTION **************
    if st.button('Analyze'):
        my_bar = st.progress(0)
        write_output()
