from huggingface_hub import InferenceClient

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful assistant for a field work vendor management team.
You have access to a vendor database containing vendor details and scoring metrics.

Guidelines:
- Answer only using the vendor records explicitly provided in the context.
- When counting vendors (e.g. "how many in X"), count every vendor record in the context that matches — do not estimate or guess.
- Your summary count must exactly match the number of vendors you list. Double-check before responding.
- When listing vendors, include all matching ones from the context — do not stop early.
- If the answer is not found in the provided context, say so clearly.
- Do not make up or infer vendor details not present in the context."""

_client: InferenceClient | None = None


def configure(hf_token: str) -> None:
    global _client
    _client = InferenceClient(model=MODEL, token=hf_token)


def get_response(
    question: str,
    context_chunks: list[str],
    history: list[dict],
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    context_text = "\n\n---\n\n".join(context_chunks)
    augmented_message = (
        f"Relevant vendor records from the database:\n\n{context_text}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": augmented_message})

    response = _client.chat_completion(messages=messages, max_tokens=MAX_TOKENS)
    return response.choices[0].message.content
