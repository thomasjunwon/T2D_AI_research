from model.diagnosing import diagnosing
from model.optimize_state import compute_total_decision
from model.progression_scoring import progression_scoring

from recommend import prepare_recommendation_items, retrieve_guidelines_for_items
from llm_writer import generate_patient_recommendation, print_recommendation
import pandas as pd
from pathlib import Path
import pickle



class PFPD():
    def __init__(self):
        

        BASE_DIR = Path(__file__).resolve().parent

        self.model_paths = [
            str(BASE_DIR / "model" / "final_checkpoints" / f"fold_{i}" / "best-checkpoint-v4.ckpt")
            for i in range(5)
        ]

        with open(BASE_DIR / "scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)


    def dbs_progression(self,X):
        X1=pd.DataFrame([X])
        result=progression_scoring(X1,self.model_paths, self.scalers)
        return result
    
    def apply_deltas(self, X, deltas):
        X_opt = X.copy()

        for col, delta in deltas.items():
            if col in X_opt.columns:
                X_opt[col] = X_opt[col] + delta

        return X_opt
    
    def lifestyle_rec(self,X):
        X3=pd.DataFrame([X])
        deltas=compute_total_decision(X3,self.model_paths,self.scalers)
        X_opt=self.apply_deltas(X3,deltas)
        final_score=progression_scoring(X_opt,self.model_paths,self.scalers)
        return deltas, final_score
    
    def make_patient_friendly_recommendation(self,X: dict, deltas: dict):
        items = prepare_recommendation_items(X, deltas, top_k=8)
        retrieved_context = retrieve_guidelines_for_items(items)
        final_json = generate_patient_recommendation(X, retrieved_context)
        print_recommendation(final_json,retrieved_context)
        return final_json, retrieved_context
    
    