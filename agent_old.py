from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

SYSTEM_PROMPT = """You are an elite football tactical analyst. Convert deviation metrics into plain-English coaching alerts.

RULES:
- 1-2 sentences only.
- Generic position names only (right back, left winger, midfield line). NEVER invent player names.
- Actionable advice: what space opened, what press stopped, what line dropped.
- Max 40 words. No fluff. No statistics. No bullet points.
- Tone: concise, authoritative, like a head coach's halftime note."""

def generate_alert(otds_score, current_minute, score_line, deviation_type):
    if otds_score < 0.4:
        return "No significant tactical deviation detected."
    
    user_msg = f"Minute {current_minute:.0f}, Score: {score_line}. Deviation: {deviation_type}. OTDS: {otds_score:.2f}."
    try:
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=60,
            temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
        words = text.split()
        if len(words) > 40:
            text = " ".join(words[:40]) + "."
        return text
    except Exception as e:
        return f"Tactical deviation detected (OTDS {otds_score:.2f}). Review {deviation_type}."
