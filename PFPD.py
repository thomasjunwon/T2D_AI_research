from model import progression_scoring, diagnosing


class PFPD():
    def __init__(self, X):
        
        X_life=X[['wk_smk',
        'wk_alc', 'wk_mvpa_work', 'wk_mvpa_play', 'wk_walk', 'wk_sleep',
        'stress', 'wk_break', 'wk_lunch', 'wk_dinner', 'wk_veg1', 'wk_veg2',
        'wk_fruit']]

        X_demo=X[['sex', 'age', 'edu', 'income', 'job']]

        X_lab=X[['chol', 'hdl', 'tg', 'ldl', 'sbp', 'wt', 'ht', 'wc', 'bmi']]

        glu=X['glu']
        hba1c=X['hba1c']
        
    def dbs_progression(X):
        result=progression_scoring(X)
        return result
    
    def dbs_diagnosis(X):
        diagnosis=diagnosing(X)
        return diagnosis
    
    def lifestyle_rec(X):
        return