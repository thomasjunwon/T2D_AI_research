import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
#client = OpenAI()


def generate_patient_recommendation(patient, retrieved_context):
    system_prompt = """
You are a medical recommendation text generator.
You do not make new medical decisions.
Use only:
1) model-derived delta values
2) retrieved ADA guideline excerpts
3) safety notes

Do not recommend medication.
Do not guarantee diabetes prevention.
Do not suggest extreme dieting, fasting, or unsafe exercise.
Write in Korean.
Use patient-friendly language.
Return JSON only.
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
                "summary": {"type": "string"},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "variable": {"type": "string"},
                            "recommendation": {"type": "string"},
                            "reason": {"type": "string"},
                            "safety_note": {"type": "string"}
                        },
                        "required": ["variable", "recommendation", "reason", "safety_note"],
                        "additionalProperties": False
                    }
                },
                "disclaimer": {"type": "string"}
            },
            "required": ["summary", "action_items", "disclaimer"],
            "additionalProperties": False
        }
    }
    """
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
    """

    #return json.loads(response.output_text)
    
    return {
    "summary":"테스트",
    "action_items":[
        {
            "variable":"wk_mvpa_play",
            "recommendation":"운동을 일주일에 150분 이상 하세요",
            "reason":"ADA guideline",
            "safety_note":"2~3일 동안 나눠서 운동하세요"
        }
    ]
    }
    
