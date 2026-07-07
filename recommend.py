def build_recommendation_item(patient: dict, var: str, delta: float):
    current_value = patient.get(var, None)
    return {
        "variable": var,
        "original_delta": delta,
        "delta": delta,
        "current_value": current_value,
    }


def prepare_recommendation_items(patient: dict, deltas: dict, top_k: int = 8):


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

        ranked.append((var, delta, ratio))

    ranked = sorted(
        ranked,
        key=lambda x: abs(x[2]),
        reverse=True
    )

    items = []

    for rank, (var, delta, ratio) in enumerate(ranked, start=1):
        item = build_recommendation_item(patient, var, delta)
        
        if item["delta"] == 0:
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
                "recommended_delta": item["delta"],
                "recommended_value": item["current_value"] + item["delta"],
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