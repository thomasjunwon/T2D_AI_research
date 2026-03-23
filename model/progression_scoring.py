def progression_scoring(X):
    import numpy as np
    import pandas as pd
    import joblib
    
    X_life=X[['wk_smk',
    'wk_alc', 'wk_mvpa_work', 'wk_mvpa_play', 'wk_walk', 'wk_sleep',
    'stress', 'wk_break', 'wk_lunch', 'wk_dinner', 'wk_veg1', 'wk_veg2',
    'wk_fruit']]

    X_demo=X[['sex', 'age', 'edu', 'income', 'job']]

    X_lab=X[['chol', 'hdl', 'tg', 'ldl', 'sbp', 'wt', 'ht', 'wc', 'bmi']]

    glu=X['glu']
    hba1c=X['hba1c']
    
    
    loaded = joblib.load("models.pkl")

    scaler = loaded["scaler"]
    model1 = loaded["model1"]
    model2 = loaded["model2"]
    model3 = loaded["model3"]
    meta_model = loaded["meta_model"]

    z1_test = model1.predict_proba(X_life) @ np.array([0,1,2])
    z2_test = model2.predict_proba(X_demo) @ np.array([0,1,2])
    z3_test = model3.predict_proba(X_lab)  @ np.array([0,1,2])


    meta_test = pd.DataFrame({
    "z1": z1_test,
    "z2": z2_test,
    "z3": z3_test,
    "glu": glu,
    "hba1c": hba1c
    })

    meta_test_scaled = scaler.transform(meta_test)

    y_proba_test = meta_model.predict_proba(meta_test_scaled)

    severity_test = y_proba_test @ np.array([0,1,2])
    severity_test*=50
    
    return severity_test
    
    
    
    