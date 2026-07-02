import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def generate_patient_recommendation(patient, retrieved_context):
    system_prompt = """
You are a medical recommendation generator for individuals with prediabetes.

Generate recommendations ONLY from:

1. model-derived lifestyle modification targets (delta values),
2. retrieved ADA guideline excerpts ("text"),
3. variable descriptions,
4. patient's current values,

Rules:
For each action item, copy the variable name exactly as provided in the recommendation context. Do not modify, rename, or omit the variable.
When you write recommendation based on delta, use the delta value of variable with patient-friendly language.
If the recommended value is not an integer, express it naturally (e.g., "about 4-5 times per week" or "approximately 8 hours") instead of reporting the exact decimal value.
Do not express activity changes as fold changes. Use absolute minutes per week and the recommended target value only.
When you write rationale, cite the reference("text") guidelines explicitly, and specifically. Do not specify where is the guideline came from (reference) when you write rationale.  
If multiple guideline texts are retrieved for the same variable, integrate all relevant information into the recommendation. Present each guideline in a separate sentence while maintaining a natural flow.
For smoking recommendations, ignore the numeric value of the adjusted delta. If the adjusted delta is negative, recommend reducing smoking and working toward complete smoking cessation.
For alcohol recommendations, do not recommend complete abstinence unless the model-derived recommended value is exactly 0 and the current intake is clearly high. Prefer wording such as "reduce alcohol intake" or "stay within recommended limits" rather than "reduce to zero cups."
"""

    user_payload = {
        "patient": patient,
        "recommendation_context": retrieved_context
    }

    schema = {
        "name": "patient_recommendation",
        "schema": {
            "type": "object",
            "properties": {
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "variable":{"type":"string"},
                            "recommendation based on delta": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["variable","recommendation based on delta", "rationale"],
                        "additionalProperties": False
                    }
                },
            },
            "required": ["action_items"],
            "additionalProperties": False
        }
    }
    
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True
            }
        }
    )
    
    return json.loads(response.output_text)


def get_references_from_context(ctx_item):
    refs = []
    for g in ctx_item.get("guidelines", []):
        meta = g.get("metadata", {})
        ref = meta.get("reference", "")
        if ref and ref not in refs:
            refs.append(ref)
    return "; ".join(refs)


def print_recommendation(result, retrieved_context):
    
    DISCLAIMER = (
    "These recommendations are generated to support lifestyle modification "
    "and do not replace professional medical advice. "
    "Please consult your healthcare provider before making major lifestyle changes."
    )

    print("=" * 80)
    print("Personalized Lifestyle Recommendation")
    print("=" * 80)

    for i, (item,ctx) in enumerate(zip(result["action_items"],retrieved_context)):
        
        meta=ctx["item"]
        
        variable=meta["variable"]
        description=meta["description"]
        current_value=meta["current_value"]
        
        reference = get_references_from_context(ctx)

        print(f"\n[{i+1}] {variable}")
        print("-" * 60)

        print(f"Variable           : {description}")
        print(f"Current value      : {current_value}")
        print(f"Recommendation     : {item['recommendation based on delta']}")
        print(f"Rationale             : {item['rationale']}")
        print(f"Reference          : {reference}")

    print("\n" + "=" * 80)
    print("Disclaimer")
    print(DISCLAIMER)
    print("=" * 80)
    
    
