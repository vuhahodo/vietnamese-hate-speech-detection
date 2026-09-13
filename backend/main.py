import os
import re
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

LABELS = ['CLEAN', 'OFFENSIVE', 'HATE']

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
WS_RE = re.compile(r'\s+')


def clean_text(s: str) -> str:
    s = URL_RE.sub(' ', s)
    s = MENTION_RE.sub(' ', s)
    s = WS_RE.sub(' ', s).strip()
    return s


class Predictor:
    """Wraps whichever model is currently available: sklearn baseline now,
    swappable for the fine-tuned PhoBERT checkpoint once Kaggle training completes
    (set MODEL_BACKEND=phobert and MODEL_DIR to the checkpoint directory)."""

    def __init__(self):
        self.backend = os.environ.get('MODEL_BACKEND', 'baseline')
        if self.backend == 'phobert':
            self._load_phobert(os.environ.get('MODEL_DIR', '/kaggle/working/phobert-vihsd-final'))
        else:
            self._load_baseline(os.environ.get('BASELINE_PATH', 'backend/model/baseline.joblib'))

    def _load_baseline(self, path):
        bundle = joblib.load(path)
        self.vectorizer = bundle['vectorizer']
        self.clf = bundle['clf']

    def _load_phobert(self, model_dir):
        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # The winning checkpoint saved by the notebook may be PhoBERT (segmented
        # training text) or XLM-R (unsegmented) -- detect from config so we don't
        # silently feed XLM-R text it never saw during training (see PORTFOLIO_ROADMAP.md).
        model_type = getattr(self.model.config, 'model_type', '') or ''
        try:
            with open(f'{model_dir}/config.json', encoding='utf-8') as f:
                model_type = json.load(f).get('model_type', model_type)
        except OSError:
            pass
        self.segment = None
        if 'xlm-roberta' not in model_type.lower() and 'xlm_roberta' not in model_type.lower():
            from pyvi import ViTokenizer
            self.segment = ViTokenizer.tokenize

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        if self.backend == 'phobert':
            text_for_model = self.segment(cleaned) if self.segment is not None else cleaned
            enc = self.tokenizer(text_for_model, truncation=True, padding=True, max_length=128, return_tensors='pt')
            with self.torch.no_grad():
                logits = self.model(**enc).logits[0]
            probs = self.torch.softmax(logits, dim=-1).numpy()
        else:
            x = self.vectorizer.transform([cleaned])
            probs = self.clf.predict_proba(x)[0]

        probs = np.asarray(probs, dtype=float)
        pred_idx = int(np.argmax(probs))
        return {
            'label': LABELS[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
            'backend': self.backend,
        }


predictor = Predictor()

app = FastAPI(title='ViHSD Hate Speech Detection API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class PredictRequest(BaseModel):
    text: str


class ProbabilityMap(BaseModel):
    CLEAN: float
    OFFENSIVE: float
    HATE: float


class PredictResponse(BaseModel):
    label: Literal['CLEAN', 'OFFENSIVE', 'HATE']
    confidence: float
    probabilities: ProbabilityMap
    backend: str


@app.get('/health')
def health():
    return {'status': 'ok', 'backend': predictor.backend}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    return predictor.predict(req.text)
