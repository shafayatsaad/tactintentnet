import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Qwen 2.5 1.5B for tactical intelligence...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print(f"Qwen loaded on {model.device}")

# Shared generation settings
GEN_KWARGS = {
    "max_new_tokens": 60,
    "temperature": 0.4,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

def _generate(prompt, max_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip().split('\n')[0].strip().replace('"', '').replace("'", "")

def generate_alert(otds_score, current_minute, score_line, deviation_type):
    if otds_score < 0.4:
        return "No significant tactical deviation detected."
    
    prompt = f"""You are an elite football assistant coach writing a halftime note.

RULES:
- One sentence only.
- Generic positions only (right back, left winger, midfield line, back line).
- Actionable advice: what space to exploit, which line to press, where to push.
- NEVER mention statistics, OTDS, or metrics.
- Max 20 words. Urgent, tactical tone. ALWAYS finish your sentence completely.

SITUATION: At minute {int(current_minute)}, opponent's {deviation_type} shifted dramatically.

COACHING NOTE:"""

    response = _generate(prompt, 40)
    if not response.endswith('.'):
        response += '.'
    words = response.split()
    if len(words) > 25:
        response = " ".join(words[:25]) + '.'
    return response

def chat_with_assistant(message, history):
    """
    Chat with the tactical AI assistant. History is list of [user, assistant] pairs.
    """
    system = """You are TactIntent, an elite football tactical analyst AI. You help managers and coaches understand match dynamics, opponent patterns, and in-game adjustments.

RULES:
- Be concise (2-3 sentences max).
- Use football terminology correctly.
- Give actionable advice, not generic platitudes.
- If asked about a specific minute, reference tactical phases (build-up, press, transition, block).
- Never hallucinate player names — use positions only."""
    
    # Build context from history (last 3 exchanges)
    context = ""
    for h in history[-3:]:
        context += f"Manager: {h[0]}\nAnalyst: {h[1]}\n"
    
    prompt = f"{system}\n\n{context}Manager: {message}\nAnalyst:"
    response = _generate(prompt, 80)
    
    # Clean up
    if "Manager:" in response:
        response = response.split("Manager:")[0].strip()
    if "Analyst:" in response:
        response = response.split("Analyst:")[-1].strip()
    
    return response

if __name__ == "__main__":
    print("Alert test:", generate_alert(0.82, 15, "2-2", "pressing structure"))
    print("Chat test:", chat_with_assistant("What should we do if they drop into a low block?", []))
