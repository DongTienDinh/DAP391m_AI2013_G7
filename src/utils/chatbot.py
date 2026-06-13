import os
import ast
import json
import pandas as pd
import streamlit as st
from pathlib import Path

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_data_context() -> str:
    """Reads file names, column schemas, and first 3 rows of CSVs in data/"""
    data_dir = PROJECT_ROOT / "data"
    context = "### Data Folder (Schemas & Snippets)\n"
    if not data_dir.exists():
        return context + "Data directory not found.\n\n"
    for ext in ["*.csv"]:
        for file in data_dir.rglob(ext):
            try:
                df = pd.read_csv(file, nrows=3)
                context += f"**File:** `{file.relative_to(PROJECT_ROOT)}`\n"
                context += f"- Columns: {list(df.columns)}\n"
                context += f"- Sample Data:\n{df.to_markdown(index=False)}\n\n"
            except Exception as e:
                context += f"Error reading {file.name}: {e}\n\n"
    return context


@st.cache_data(show_spinner=False)
def load_reports_context() -> str:
    """Reads text-based reports, leaderboards, and logs directly."""
    reports_dir = PROJECT_ROOT / "reports"
    outputs_dir = PROJECT_ROOT / "outputs"
    context = "### Reports and Outputs\n"
    dirs_to_check = [reports_dir, outputs_dir]
    valid_exts = [".csv", ".txt", ".md", ".json"]
    for d in dirs_to_check:
        if not d.exists():
            continue
        for file in d.rglob("*"):
            if file.is_file() and file.suffix in valid_exts:
                try:
                    context += f"**File:** `{file.relative_to(PROJECT_ROOT)}`\n"
                    if file.suffix == ".csv":
                        df = pd.read_csv(file, nrows=50)
                        context += f"{df.to_markdown(index=False)}\n\n"
                    else:
                        with open(file, encoding="utf-8") as f:
                            content = f.read()
                        if len(content) > 3000:
                            content = content[:3000] + "\n...[TRUNCATED]"
                        context += f"{content}\n\n"
                except Exception as e:
                    context += f"Error reading {file.name}: {e}\n\n"
    return context


@st.cache_data(show_spinner=False)
def load_source_code_context() -> str:
    """Traverses src/ to extract class/function signatures and docstrings via AST parsing."""
    src_dir = PROJECT_ROOT / "src"
    context = "### Source Code (Signatures & Docstrings)\n"
    if not src_dir.exists():
        return context + "Src directory not found.\n\n"
    for file in src_dir.rglob("*.py"):
        try:
            with open(file, encoding="utf-8") as f:
                node = ast.parse(f.read())
            context += f"**File:** `{file.relative_to(PROJECT_ROOT)}`\n"
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    doc = ast.get_docstring(item)
                    doc_str = f' — Docstring: """{doc}"""' if doc else ""
                    context += f"- Function: `def {item.name}(...)`{doc_str}\n"
                elif isinstance(item, ast.ClassDef):
                    doc = ast.get_docstring(item)
                    doc_str = f' — Docstring: """{doc}"""' if doc else ""
                    context += f"- Class: `class {item.name}`{doc_str}\n"
            context += "\n"
        except Exception as e:
            context += f"Error parsing {file.name}: {e}\n\n"
    return context


@st.cache_data(show_spinner="Loading context (this runs once per session)...")
def build_project_context() -> str:
    """Aggregates all project context."""
    context = "Here is the current state and structure of the project:\n\n"
    context += load_data_context()
    context += load_reports_context()
    context += load_source_code_context()
    return context


def _call_gemini(prompt: str, system_instruction: str, api_key: str) -> str | None:
    """Call Gemini using google.genai client."""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[system_instruction, prompt],
        )
        return response.text if response else None
    except Exception:
        return None


def init_chatbot():
    """Initializes chat session. Call once at app start."""
    api_key = os.getenv("GEMINI_API_KEY")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_api_key" not in st.session_state:
        st.session_state.chat_api_key = api_key
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = build_project_context() if api_key else ""
    return bool(api_key)


def render_chatbot_ui():
    """Renders the chatbot UI inside whatever container it is called in."""
    st.markdown("### 🤖 AI Project Assistant")
    st.markdown("Ask anything about the project's data, source code, outputs, and reports!")

    if not st.session_state.chat_api_key:
        st.warning("GEMINI_API_KEY not set. Chat unavailable.")
        return

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    system_instruction = (
        "You are an AI assistant for this specific Data Science project. "
        "Use the provided context to answer user questions. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"{st.session_state.chat_context}"
    )

    main_container = st.container(height=600, border=False)

    with main_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    text = _call_gemini(prompt, system_instruction, st.session_state.chat_api_key)
                    if text:
                        st.markdown(text)
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.messages.append({"role": "assistant", "content": text})
                    else:
                        st.error("Failed to get response from Gemini.")
            st.session_state.pending_prompt = None

        if prompt := st.chat_input("Ask a question about the project..."):
            st.session_state.pending_prompt = prompt
            st.rerun()
