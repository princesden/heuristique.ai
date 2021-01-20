import uvicorn
import transformers
import torch

from fastapi import FastAPI

# ML Package
import joblib
import os

# Vectorized
with open('zero_shot_model.pkl', 'rb') as file:
    facebook_bert_zero_shot = joblib.load(file)

# init app
app = FastAPI()


@app.get('/predict/{sequence_to_classify}')
async def predict(sequence_to_classify):
    candidate_labels = ['Availability heuristic', 'Action Bias', 'Decision Fatigue', 'Halo Effect', 'Pessimism bias',
                        'Social Norms']
    return facebook_bert_zero_shot(sequence_to_classify, candidate_labels, multi_class=True)


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)
