import streamlit as st
import os
import re
from dotenv import load_dotenv

# ===== IMPORTS =====
# Kalau env lama, pakai import langsung dari langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ========================
# Load env & constants
# ========================
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL")
EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL")
FAISS_INDEX_DIR= os.getenv("FAISS_INDEX_DIR")

# ========================
# Load FAISS index
# ========================
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embedder, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ========================
# LLM & Prompt
# ========================
SYSTEM_PROMPT = (
    "You are a helpful and concise customer assistant.\n"
    "Always answer ONLY using the CONTEXT below.\n"
    "If the answer is not found in the context, say you don't know.\n"
    "Always include the source (filename + row + chunk_id)."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer in English.")
])

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# ========================
# Build QA chain
# ========================
def qa_chain(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    # Hapus semua teks dalam tanda kurung
    text = re.sub(r"\(.*?\)", "", text).strip()
    return text

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Customer Assistant", page_icon="🤖")
st.title("🤖 Intelligent Customer Assistant")

st.markdown("Ask a question based on the dataset, and the assistant will answer with retrieved context.")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_q = st.chat_input("Type your question here...")
if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})
    answer = qa_chain(user_q)
    st.session_state.chat.append({"role": "assistant", "content": answer})

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
