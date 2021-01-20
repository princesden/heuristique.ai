import uvicorn
import transformers
import torch
import joblib
import os
from fastapi import FastAPI

# init app
app = FastAPI()

# Vectorized

with open('models/zero_shot_model.pkl', 'rb') as file:
    facebook_bert_zero_shot = joblib.load(file)

with open('models/valhalla-distilbart-mnli-12-6.pkl', 'rb') as file:
    valhalla_distilbart_mnli = joblib.load(file)


@app.get('/')
def load_page():
    return {"text": "Welcome to Heuristique.ai "}


@app.get('/predict/v1/{sequence_to_classify}')
async def predict_v1(sequence_to_classify):
    candidate_labels = ['Availability heuristic', 'Action Bias', 'Decision Fatigue', 'Halo Effect', 'Pessimism bias',
                        'Social Norms']
    return facebook_bert_zero_shot(sequence_to_classify, candidate_labels, multi_class=True)


@app.get('/predict/v2/{sequence_to_classify}')
async def predict_v2(sequence_to_classify):
    candidate_labels = ['Availability heuristic', 'Action Bias', 'Decision Fatigue', 'Halo Effect', 'Pessimism bias',
                        'Social Norms']
    return valhalla_distilbart_mnli(sequence_to_classify, candidate_labels, multi_class=True)

# TO run >>  uvicorn main:app --reload

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)
