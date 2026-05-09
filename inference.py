import numpy as np
from sklearn.mixture import GaussianMixture
from collections import deque

class TacticalFingerprintGMM:
    def __init__(self, n_components=8, random_state=42):
        self.gmm = GaussianMixture(
            n_components=n_components, covariance_type='full',
            random_state=random_state, max_iter=200
        )
        self.fitted = False
    
    def fit(self, embeddings):
        X = np.stack(embeddings, axis=0)
        self.gmm.fit(X)
        self.fitted = True
        scores = self.gmm.score_samples(X)
        self.min_score = float(scores.min())
        self.max_score = float(scores.max())
        return self
    
    def score(self, embedding):
        if not self.fitted:
            raise RuntimeError("GMM not fitted")
        emb = embedding.reshape(1, -1)
        log_lik = self.gmm.score_samples(emb)[0]
        clamped = np.clip(log_lik, self.min_score, self.max_score)
        normalized = (clamped - self.min_score) / (self.max_score - self.min_score + 1e-8)
        return float(1.0 - normalized)

class OTDSCalculator:
    def __init__(self, gmm, window_size=15):
        self.gmm = gmm
        self.window = deque(maxlen=window_size)
        self.otds_history = []
    
    def update(self, embedding, minute):
        self.window.append(embedding)
        if len(self.window) < 3:
            return 0.0
        scores = [self.gmm.score(e) for e in self.window]
        avg = np.mean(scores)
        if self.otds_history:
            avg = 0.7 * avg + 0.3 * self.otds_history[-1][1]
        self.otds_history.append((minute, avg))
        return avg
    
    def detect_spike(self, threshold=0.6):
        if not self.otds_history:
            return False, 0.0
        latest = self.otds_history[-1][1]
        return latest > threshold, latest
    
    def get_timeline(self):
        if not self.otds_history:
            return [], []
        mins, scores = zip(*self.otds_history)
        return list(mins), list(scores)
