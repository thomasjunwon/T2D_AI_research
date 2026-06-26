from PFPD import PFPD
import pandas as pd


X = {
    "sex": 2.0,
    "age": 54.0,
    "edu": 4.0,
    "income": 3.0,
    "job": 2.0,
    "glu": 104.0,
    "hba1c": 5.2,
    "chol": 179.0,
    "hdl": 85.0,
    "tg": 60.0,
    "ldl": 90.0,
    "sbp": 103.0,
    "wt": 60.7,
    "ht": 169.2,
    "wc": 69.8,
    "bmi": 21.202544,

    "wk_smk": 0.0,
    "wk_alc": 0.4375,
    "wk_mvpa_play": 0.0,
    "wk_walk": 360.0,
    "wk_sleep": 360.0,
    "stress": 3.0,
    "wk_break": 0.0,
    "wk_lunch": 6.0,
    "wk_dinner": 3.5,
    "wk_veg1": 7.0,
    "wk_veg2": 3.0,
    "wk_fruit": 5.5
}
#stress: 스트레스를 평소에, 대단히 많이 느낀다/많이 느끼는 편이다/조금 느끼는 편이다/거의 느끼지 않는다.

pro=PFPD()

print(f"current score: {pro.dbs_progression(X)}")
deltas, final_score = pro.lifestyle_rec(X)

print(f"recommend: {deltas}")
print(f"final score: {final_score}")

print(f"guideline: {pro.make_patient_friendly_recommendation(X,deltas)}")


