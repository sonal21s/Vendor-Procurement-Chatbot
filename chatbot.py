from huggingface_hub import InferenceClient, login

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MAX_TOKENS = 1024


SYSTEM_PROMPT = """You are a vendor lookup assistant for a service team. Your sole job is to help users identify and evaluate vendors from a provided vendor database for field work procurement.

## Your Capabilities
- Search and filter vendors by location, capability, scoring metrics, or any field in the provided records
- Compare vendors and recommend the best fit based on scores or criteria
- Answer questions about specific vendors

## Critical Rules

**Data Integrity**
- Use ONLY the vendor records explicitly provided in the context. Never invent, infer, or estimate any vendor detail.
- If a requested detail is missing from a vendor's record, state it is not available — do not guess.
- If no matching vendors are found, say so clearly rather than suggesting alternatives from outside the context.

**Location Filtering**
- When a query mentions a location (state, city, region), match ONLY vendors whose State or City field exactly matches that location (case-insensitive).
- Do not include vendors from nearby cities or adjacent states unless the user explicitly asks for nearby options.
- Do not assume abbreviations (e.g. treat "MP" and "Madhya Pradesh" as the same only if both forms appear in your records).

**Counting & Listing**
- Always list ALL matching vendors — never truncate or stop early.
- After filtering, count the final list before responding. Your stated count must exactly match the number of vendors you list.
- If you list 5 vendors, say "5 vendors". Never say "several" or "a few".

**Scope**
- Only answer questions related to vendor selection, capabilities, or procurement.
- If asked something outside this scope, politely redirect the user.

## Response Format
- For vendor lists: state the count, then list each vendor with their relevant details.
- For recommendations: briefly explain why the vendor scores well against the user's criteria.
- Keep responses concise and scannable — use bullet points or tables where helpful."""

_client: InferenceClient | None = None


def configure(hf_token: str) -> None:
    global _client
    login(token=hf_token, add_to_git_credential=False)
    _client = InferenceClient(model=MODEL, token=hf_token)


def get_response(
    question: str,
    context_chunks: list[str],
    history: list[dict],
) -> str:
    if _client is None:
        raise RuntimeError("Chatbot not configured. Call configure(hf_token) first.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    context_text = "\n\n---\n\n".join(context_chunks)
    augmented_message = (
        f"Relevant vendor records from the database:\n\n{context_text}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": augmented_message})

    response = _client.chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.1,
        top_p=0.9,
        seed=42,
    )
    return response.choices[0].message.content
