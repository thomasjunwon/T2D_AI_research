# recommend.py
"""
VARIABLE_MAP = {

    "wk_walk": "physical_activity",
    "wk_mvpa_play": "physical_activity",

    "wk_break": "diet",
    "wk_lunch": "diet",
    "wk_dinner": "diet",
    "wk_veg1": "diet",
    "wk_veg2": "diet",
    "fruit": "diet",

    "wk_smk": "smoking",

    "wk_alc": "alcohol",

    "stress": "stress",

    "wk_sleep" : "sleep"
}


def map_variable_to_group(var_name: str) -> str: #delta 변수 이름 입력 -> 변수의 group 출력
    name = var_name.lower()

    for key, group in VARIABLE_MAP.items():
        if key in name:
            return group

    return "general"
"""

# recommend.py

def safety_adjust_delta(patient: dict, var: str, delta: float):
    current_value = patient.get(var, None)
    #group = map_variable_to_group(variable)

    adjusted_delta = delta
    safety_note = ""

    # 운동: 과도한 증가량 완화
    if var == "wk_mvpa_play":
        if delta > 150:
            adjusted_delta = 150
            safety_note = "The recommended increase in physical activity has been adjusted to promote gradual progression toward 150 minutes per week."

    # 흡연: delta 숫자보다 금연 권고로 표현
    if var == "smoking":
        safety_note = "Smoking-related recommendations focus on smoking cessation or reduction rather than a specific quantitative target."
    
    if var=="stress":
        if delta < 1:
            stress_level = "slight"

        elif delta < 2:
            stress_level = "moderate"

        else:
            stress_level = "high"

        safety_note=f"Recommended adjustment intensity: {stress_level}"

    return {
        "variable": var,
        "original_delta": delta,
        "adjusted_delta": adjusted_delta,
        "current_value": current_value,
        "safety_note": safety_note
    }


def prepare_recommendation_items(patient: dict, deltas: dict, top_k: int = 8): 
    items = []  #리스트 안에 delta 변수들에 note랑 original value 달아서 저장


    for var, val in list(deltas.items())[:top_k]:
        delta = val[0]

        item = safety_adjust_delta(patient, var, float(delta))
        items.append(item)

    return items

# recommend.py

import chromadb


def get_ada_collection(persist_path="./chroma_dia"): #chroma_ada에서 가이드라인 client 가져온다
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(name="dia_guidelines")


def retrieve_guidelines_for_items(items, persist_path="./chroma_dia", n_results=2):
    collection = get_ada_collection(persist_path)

    retrieved = []

    for item in items:
        var = item["variable"]

        query_text = f"Clinical guidelines for {var} in prediabetes prevention"

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"variable": var} if var != "general" else None
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        retrieved.append({
            "item": item,
            "guidelines": [
                {
                    "text": doc,
                    "metadata": meta
                }
                for doc, meta in zip(docs, metas)
            ]
        })

    return retrieved