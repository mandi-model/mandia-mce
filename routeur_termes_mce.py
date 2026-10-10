"""Restore the experimental closed-set civil/penal term router from JSON.
Only takes the given term, not its definition. No general/OOD detector.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class RouteurTermesMCE:
    def __init__(self,path,expected_sha256=None):
        raw=Path(path).read_bytes()
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('Router fingerprint does not match the manifest')
        state=json.loads(raw)
        if state['classes']!=['civil','penal']:raise ValueError('Unexpected router classes')
        vocab=state['vocabulary'];self.coef=np.asarray(state['coef'],dtype=np.float64)
        idf=np.asarray(state['idf'],dtype=np.float64);self.intercept=float(state['intercept'])
        if sorted(vocab.values())!=list(range(len(vocab))):raise ValueError('Invalid vocabulary indices')
        if self.coef.shape!=(len(vocab),) or idf.shape!=self.coef.shape:raise ValueError('Invalid router dimensions')
        if not np.isfinite(self.coef).all() or not np.isfinite(idf).all() or not np.isfinite(self.intercept):
            raise ValueError('Nonfinite router weights')
        self.vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),vocabulary=vocab,dtype=np.float64)
        self.vectorizer.idf_=idf
        self.state=state

    def choisir(self,terme):
        if not isinstance(terme,str) or not terme.strip() or '\n' in terme or '\r' in terme:
            raise ValueError('Provide one nonempty legal term without its definition')
        score=float(np.asarray(self.vectorizer.transform([terme])@self.coef).ravel()[0]+self.intercept)
        return self.state['classes'][int(score>0)]

    def choisir_plusieurs(self,termes):
        return [self.choisir(terme) for terme in termes]
