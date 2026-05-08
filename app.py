import streamlit as st
import chatbot
from sheets import load_sheet_data, records_to_chunks
from vectorstore import VectorStore

st.set_page_config(
    page_title="Vendor Assistant",
    page_icon="📋",
    layout="centered",
)

# ── Secrets ──────────────────────────────────────────────────────────────────
SHEET_ID = st.secrets["SHEET_ID"]
HF_TOKEN = st.secrets["HF_TOKEN"]

if "chatbot_configured" not in st.session_state:
    chatbot.configure(HF_TOKEN)
    st.session_state["chatbot_configured"] = True


# ── Data loading & indexing (cached for the lifetime of the server instance) ─
@st.cache_resource(show_spinner="Loading and indexing vendor database...")
def build_vectorstore() -> tuple[VectorStore, int]:
    records = load_sheet_data(SHEET_ID)
    if not records:
        st.error("The Google Sheet appears to be empty.")
        st.stop()
    chunks = records_to_chunks(records)
    vs = VectorStore()
    vs.index(chunks)
    return vs, len(records)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Vendor Assistant")
    st.markdown("Ask anything about your vendor database.")
    st.divider()

    vs, vendor_count = build_vectorstore()
    st.metric("Vendors indexed", vendor_count)

    if st.button("Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Chat UI ───────────────────────────────────────────────────────────────────
st.header("Vendor Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask about vendors, scores, contacts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vendor database..."):
            context_chunks = vs.query(prompt, n_results=5)
            # Pass history excluding the current user message
            history = st.session_state.messages[:-1]
            response = chatbot.get_response(prompt, context_chunks, history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
