import streamlit as st
from annotated_text import annotated_text
import nltk
import joblib

#              PAGE CONFIG
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


# ****************SIDEBAR / CONFIGURATIONS*********

st.sidebar.image('logo.png', width=300)
#bias_selection = st.sidebar.multiselect('Select biases', ["Action Bias", "Availability Heuristics", "Decision Bias"])
bias_selection = st.sidebar.multiselect('Select biases', ["Appeal to Probability", "Bad Reasons Fallacy", "Masked Man Fallacy", "Non Sequitur"])
output_sensitivity = st.sidebar.select_slider('Algorithm Sensitivity', options=[50, 60, 70, 80, 90, 100])
is_multiple_correct_answers = st.sidebar.checkbox('Multiple correct answers')
analysis_boundary = st.sidebar.radio('Analysis Boundary', ["Sentence", "Corpus"])
colors = {bias_selection[0]: '#8ef', bias_selection[1]: '#faa', bias_selection[2]: '#afa', bias_selection[3]: '#fea'}


def configure_individual_bias():
    is_configure_individual_bias = st.sidebar.checkbox('Configure Biases Individually')
    return is_configure_individual_bias


def pick_classification_color():
    color_pick = st.sidebar.color_picker('Pick output color for each Bias')
    return color_pick


def bias_wiki():
    st.sidebar.radio('Wiki', ['Definitions'])
    # #st.sidebar.error('What is the Action Bias? The action bias describes our tendency to favor '
    #                                           'action over inaction, often to our benefit. However, there are times when'
    #                                           ' we feel compelled to act, even if theres no evidence that it will lead '
    #                                           'to a better outcome than doing nothing would.')


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


def write_output():
    result = []
    prediction_set = execute_ml()
    for prediction in prediction_set:
        # st.json(prediction)
        result.append(construct_annotation_elements(prediction))
    annotated_text(*result, height=1000)


st.title('Sentence Bias Analysis')
st.write('Heuristics are the strategies derived from previous experiences with similar problems. These strategies '
         'depend on using readily accessible, though  applicable, information to control problem solving in '
         'human beings, machines and abstract issues.')
input_text = st.text_area('Enter Text', height=200, max_chars=10000)

if __name__ == '__main__':
    if st.button('Analyze'):
        write_output()

