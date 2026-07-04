def safety_adjust_delta(patient: dict, var: str, delta: float):
    current_value = patient.get(var, None)

    adjusted_delta = delta

    if var == "wk_mvpa_play":
        if delta < 0 or current_value >= 150:
            adjusted_delta = 0
    
    if var == "wk_walk":
        if delta < 0 or current_value >= 420:  #근거: 하루에 1시간 까지가 이점이 존재한다
            adjusted_delta = 0
    
    if var in ["wk_break", "wk_lunch", "wk_dinner", "wk_fruit"]:
        if delta < 0:
            adjusted_delta = 0 
            
    if var == "wk_fruit":
        if current_value >= 14:        #7번보다 많이 먹으면 안된다는 근거?
            adjusted_delta = 0
    
    if var in ["wk_veg1", "wk_veg2"]:
        if current_value >= 14:
            adjusted_delta = 0
    
    if var =="wk_alc":
        if abs(delta) < 1:
            adjusted_delta=0
            
    if var == "wk_sleep":
        if current_value < 360:
            adjusted_delta = min(60, 360 - current_value)
        elif current_value > 540:
            adjusted_delta = max(-60, 540 - current_value)
        elif 360 <= current_value <= 540:
            adjusted_delta = 0
            
    if var == "wk_smk":
        if delta < 0:
            adjusted_delta = -1 
        else:
            adjusted_delta = 0
    
    return {
        "variable": var,
        "original_delta": delta,
        "adjusted_delta": adjusted_delta,
        "current_value": current_value,
    }


def prepare_recommendation_items(patient: dict, deltas: dict, top_k: int = 8, min_abs_ratio: float = 0.01):
    """
    deltas: {"wk_mvpa_play": 50.4, "stress": -2.0, ...}
    top_k: normalized absolute delta 기준 상위 변수 개수
    min_abs_ratio: 너무 작은 변화량 제거 기준
    """

    inv = {
        'wk_smk': (0.0, 420.0),
        'wk_alc': (0.0, 40.0),
        'wk_mvpa_play': (0.0, 300.0),
        'wk_walk': (0.0, 1260.0),
        'wk_sleep': (360.0, 540.0),
        'stress': (1.0, 4.0),
        'wk_break': (0.0, 6.0),
        'wk_lunch': (0.0, 6.0),
        'wk_dinner': (0.0, 6.0),
        'wk_veg1': (0.0, 21.0),
        'wk_veg2': (0.0, 21.0),
        'wk_fruit': (0.0, 21.0),
    }

    ranked = []

    for var, delta in deltas.items():

        delta = float(delta)
        min_val, max_val = inv[var]
        scale = max_val - min_val

        ratio = delta / scale

        # 너무 작은 변화는 제외
        if abs(ratio) < min_abs_ratio:
            continue

        ranked.append((var, delta, ratio))

    ranked = sorted(
        ranked,
        key=lambda x: abs(x[2]),
        reverse=True
    )

    items = []

    for rank, (var, delta, ratio) in enumerate(ranked, start=1):
        item = safety_adjust_delta(patient, var, delta)
        
        if item["adjusted_delta"] == 0:
            continue
    
        item["ratio"] = ratio
        item["rank"] = rank
        items.append(item)
        
        if len(items)>=top_k:
            break

    return items



import chromadb


def get_ada_collection(persist_path="./chroma_dia"): #chroma_ada에서 가이드라인 client 가져온다
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(name="dia_guidelines")


def retrieve_guidelines_for_items(items, persist_path="./chroma_dia", n_results=3):
    collection = get_ada_collection(persist_path)

    retrieved = []
    VARIABLE_DESCRIPTION = {
    "wk_break": "Number of days you eat breakfast each week",
    "wk_lunch": "Number of days you eat lunch each week",
    "wk_dinner": "Number of days you eat dinner each week",

    "wk_mvpa_play": "Weekly minutes of moderate-to-vigorous physical activity",

    "wk_walk": "Weekly minutes of walking",

    "wk_sleep": "Average daily sleep duration (minutes)",

    "wk_fruit": "Weekly frequency of fruit consumption",

    "wk_veg1": "Weekly frequency of consuming vegetables, mushrooms, and seaweed, including kimchi and pickled vegetables",

    "wk_veg2": "Weekly frequency of consuming vegetables, mushrooms, and seaweed, excluding kimchi and pickled vegetables",

    "stress": "Self-reported stress level, from 1 to 4",
    
    "wk_alc": "Weekly alcohol consumption (cups per week)",
    
    "wk_smk": "Weekly frequency of smoking"
    }

    for item in items:
        var = item["variable"]

        query_text = VARIABLE_DESCRIPTION[item["variable"]]

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"variable": var} if var != "general" else None
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        retrieved.append({
            "item": {
                "variable": item["variable"],
                "description": VARIABLE_DESCRIPTION[item["variable"]],
                "current_value": item["current_value"],
                "recommended_delta": item["adjusted_delta"],
                "recommended_value": item["current_value"] + item["adjusted_delta"],
            },
            "guidelines": [
                {
                    "text": doc,
                    "metadata": meta
                }
                for doc, meta in zip(docs, metas)
            ]
        })

    return retrieved