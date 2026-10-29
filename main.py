"""Streamlit interface for asking questions about uploaded documents."""

from __future__ import annotations

import hashlib
import io

import streamlit as st
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()


def apply_dashboard_theme() -> None:
    """Apply the dark pink/rose dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --pink: #ec4899;
            --pink-bright: #f472b6;
            --pink-dark: #be185d;
            --surface: #181115;
            --surface-raised: #22171d;
            --border: #3d2632;
            --muted: #b89ea9;
        }

        .stApp {
            color: #fdf2f8;
            background:
                radial-gradient(circle at 85% 0%, rgba(236, 72, 153, 0.22), transparent 32rem),
                radial-gradient(circle at 10% 100%, rgba(190, 24, 93, 0.18), transparent 28rem),
                #0e090c;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #191015 0%, #110b0e 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: #f5d0fe;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        .rag-hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
            padding: 1.7rem 2rem;
            border: 1px solid #50263b;
            border-radius: 18px;
            background: linear-gradient(115deg, rgba(48, 20, 35, .96), rgba(22, 13, 18, .96));
            box-shadow: 0 18px 50px rgba(0, 0, 0, .35);
        }

        .rag-hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            right: -80px;
            top: -150px;
            border-radius: 50%;
            background: rgba(236, 72, 153, .22);
            filter: blur(22px);
        }

        .rag-eyebrow {
            margin-bottom: .45rem;
            color: var(--pink-bright);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .rag-title {
            margin: 0;
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.25rem);
            font-weight: 850;
            letter-spacing: -.045em;
            line-height: 1;
        }

        .rag-title span { color: var(--pink-bright); }

        .rag-subtitle {
            max-width: 690px;
            margin: .85rem 0 0;
            color: var(--muted);
            font-size: 1rem;
        }

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stChatMessage"],
        div[data-testid="stAlert"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(30, 19, 25, .90);
        }

        [data-testid="stChatMessage"] {
            margin-bottom: .7rem;
            padding: .35rem .45rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid var(--pink);
        }

        .stButton > button[kind="primary"] {
            border: 1px solid #f472b6;
            background: linear-gradient(135deg, #ec4899, #be185d);
            color: white;
            font-weight: 750;
            box-shadow: 0 8px 24px rgba(236, 72, 153, .25);
        }

        .stButton > button[kind="primary"]:hover {
            border-color: #f9a8d4;
            background: linear-gradient(135deg, #f472b6, #db2777);
        }

        .stButton > button[kind="secondary"] {
            border-color: var(--border);
            background: var(--surface-raised);
            color: #fce7f3;
        }

        [data-testid="stChatInput"] {
            border: 1px solid #592b42;
            border-radius: 14px;
            background: #1c1217;
            box-shadow: 0 10px 35px rgba(0, 0, 0, .28);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--pink);
            box-shadow: 0 0 0 2px rgba(236, 72, 153, .18);
        }

        [data-testid="stFileUploaderDropzone"] button {
            border-color: #9d174d;
            background: #381526;
            color: #fce7f3;
        }

        h1, h2, h3 { letter-spacing: -.025em; }
        a { color: var(--pink-bright) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def uploaded_file_to_document(uploaded_file) -> Document:
    """Extract text from a supported Streamlit UploadedFile."""
    raw_data = uploaded_file.getvalue()
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(raw_data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw_data.decode("utf-8", errors="replace")

    if not text.strip():
        raise ValueError(f"No readable text was found in {uploaded_file.name}.")

    return Document(page_content=text, metadata={"source": uploaded_file.name})


def file_set_id(uploaded_files) -> str:
    """Return a stable identifier for the currently uploaded file contents."""
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def build_agent(documents: list[Document]):
    """Build a session-local vector index and Deep Agent optimized for Free Tier."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vector_store.add_documents(chunks)

    backend = StateBackend()

    @tool(parse_docstring=True)
    def search_documentation(query: str) -> str:
        """Search uploaded documents and return matching chunks directly.

        Args:
            query: Natural-language search query.

        Returns:
            Retrieved document chunks with their sources.
        """
        retrieved_docs = vector_store.similarity_search(query, k=min(3, len(chunks)))
        results = []
        for index, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"--- Chunk {index} (Source: {source}) ---\n{doc.page_content}")

        return "\n\n".join(results)

    simple_instructions = """You are a helpful document Q&A assistant.
1. Always call `search_documentation` first to find relevant information from the uploaded documents.
2. Answer the user's question clearly and accurately based ONLY on the retrieved chunks.
3. Mention the source file name in your answer.
4. Do not answer from memory if the information is not in the documents."""

    model = init_chat_model(
       model="google_genai:gemini-3.5-flash-lite",
        max_retries=5,
    )

    agent = create_deep_agent(
        model=model,
        tools=[search_documentation],
        backend=backend,
        system_prompt=simple_instructions,
    )
    return agent, len(chunks)


def final_text(result: dict) -> str:
    """Extract the last readable model message from an agent result."""
    for message in reversed(result.get("messages", [])):
        text = getattr(message, "text", "") or getattr(message, "content", "")
        if isinstance(text, str) and text.strip():
            return text
        if isinstance(text, list):
            parts = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in text
            ]
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                return joined
    return "The agent completed the request but did not return a text response."


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon="🌸", layout="wide")
    apply_dashboard_theme()
    st.markdown(
        """
        <section class="rag-hero">
            <div class="rag-eyebrow">LangChain · Retrieval augmented generation</div>
            <h1 class="rag-title">Chat with your <span>knowledge</span></h1>
            <p class="rag-subtitle">
                Upload your documents, build a semantic index, and get grounded answers
                from an agent that searches before it speaks.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("### 🌸 Knowledge base")
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        process_clicked = st.button(
            "Process documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

        if process_clicked:
            try:
                current_file_set = file_set_id(uploaded_files)
                with st.spinner("Reading, splitting, and indexing documents..."):
                    documents = [uploaded_file_to_document(item) for item in uploaded_files]
                    agent, chunk_count = build_agent(documents)
                st.session_state.agent = agent
                st.session_state.file_set_id = current_file_set
                st.session_state.messages = []
                st.success(f"Indexed {len(documents)} files into {chunk_count} chunks.")
            except Exception as exc:
                st.error(f"Could not process the documents: {exc}")

        if uploaded_files and "file_set_id" in st.session_state:
            if file_set_id(uploaded_files) != st.session_state.file_set_id:
                st.warning("The selected files changed. Process them before asking questions.")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your documents",
        disabled="agent" not in st.session_state,
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing your documents..."):
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=question)]}
                    )
                    answer = final_text(result)
                except Exception as exc:
                    answer = f"I couldn't answer that question: {exc}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if "agent" not in st.session_state:
        st.info("Upload and process at least one document to begin.")


if __name__ == "__main__":
    main()