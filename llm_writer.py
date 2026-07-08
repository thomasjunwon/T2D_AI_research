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
When writing the rationale, base every statement explicitly on the retrieved guideline text. Do not introduce unsupported medical claims or cite the guideline source by name.
If multiple guideline texts are retrieved for the same variable, integrate all relevant information into the recommendation. Present each guideline in a separate sentence while maintaining a natural flow.
For smoking recommendations, ignore the numeric value of the delta. If the delta is negative, recommend reducing smoking and working toward complete smoking cessation.
For alcohol recommendations, avoid recommending complete abstinence unless clearly warranted. Prefer wording such as "reduce alcohol intake" or "stay within recommended limits" over rigid numerical targets whenever appropriate.
For walking recommendations:If the recommended walking target is substantially higher than the patient's current walking time, recommend increasing walking gradually rather than immediately reaching the target.

For meal-related variables (breakfast, lunch, and dinner), never recommend skipping regular meals or reducing meal frequency to clinically unreasonable levels. If the model-derived target suggests reducing meal frequency, reinterpret the recommendation using the retrieved guideline content rather than explicitly recommending fewer meals.
For fruit intake, never recommend eliminating fruit consumption. If the model-derived target suggests a lower intake, express the recommendation in accordance with the retrieved guideline content rather than avoiding fruit entirely.
Avoid recommending target values that are unrealistic for immediate implementation. When the suggested increase is large, describe it as a gradual progression toward the target rather than an immediate change.
When the model-derived target and the retrieved guideline appear inconsistent, prioritize the retrieved guideline when generating patient-facing recommendations while preserving the overall intended direction of lifestyle modification.
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
    
    
