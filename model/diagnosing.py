def diagnosing(X):
    X_cause=X[['wk_smk',
    'wk_alc', 'wk_mvpa_work', 'wk_mvpa_play', 'wk_walk', 'wk_sleep',
    'stress', 'wk_break', 'wk_lunch', 'wk_dinner', 'wk_veg1', 'wk_veg2',
    'wk_fruit','sex', 'age', 'edu', 'income', 'job']]


    X_result=X[['chol', 'hdl', 'tg', 'ldl', 'sbp', 'wt', 'ht', 'wc', 'bmi','glu','hba1c']]

    import numpy as np
    import pandas as pd
    import joblib

    def temperature_scaling(probs, T=2.0):
        # 안정성 위해 log-space 처리
        log_probs = np.log(probs + 1e-12)
        scaled_log_probs = log_probs / T
        scaled_probs = np.exp(scaled_log_probs)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
        return scaled_probs
    
    loaded = joblib.load("models2.pkl")

    scaler = loaded["scaler"]
    model1 = loaded["model1"]
    model2 = loaded["model2"]
    meta_model = loaded["meta_model"]
    
    z1_test = model1.predict_proba(X_cause) @ np.array([0,1,2])
    z2_test = model2.predict_proba(X_result) @ np.array([0,1,2])
    
    meta_test = pd.DataFrame({
    "z1": z1_test,
    "z2": z2_test,
    })

    meta_test_scaled = scaler.transform(meta_test)

    y_proba_test = meta_model.predict_proba(meta_test_scaled)

    y_proba_test = temperature_scaling(y_proba_test, T=4)

    diagnosis = y_proba_test @ np.array([0,1,2])
    
    return diagnosis

    
