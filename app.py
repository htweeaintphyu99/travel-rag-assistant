"""
Streamlit UI for the travel assistant's RAG pipeline.

Wraps rag_pipeline.rag() in a chat interface, with a sidebar for
(re)building the Elasticsearch knowledge base — ingestion/indexing is
slow (fetches + embeds + indexes wiki articles), so it's kept as an
explicit, separate action instead of running on every query like the
current main.py does.

Setup:
    uv add streamlit

Usage:
    uv run streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from ingest import ingest
from index import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX, run
from search_engine import SearchEngine
from travel_assistant.rag_pipeline import NATURAL_PROMPT_TEMPLATE, rag, to_log_record, evaluate_relevance
from db.db_init import *
from db.db_save import save_conversation
from db.db_feedback import save_feedback
from dotenv import load_dotenv

load_dotenv()

CHUNK_PATH = "data/chunks.json"
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 150
MODEL = "gemini-3.5-flash"
ESL_URL = os.getenv( "ES_URL", "http://localhost:9200")

st.set_page_config(
    page_title="Travel Assistant", page_icon="\U0001f9f3", layout="centered"
)


@st.cache_resource
def get_search_engine() -> SearchEngine:
    return SearchEngine(host=ESL_URL, index_name=DEFAULT_INDEX)


@st.cache_resource
def init_db_and_feedback() -> None:
    init_db()
    init_feedback()
    print("Database initialized")


def doc_count(engine: SearchEngine) -> int:
    try:
        return engine.client.count(index=DEFAULT_INDEX)["count"]
    except Exception:
        return 0


def build_knowledge_base(
    country: str, max_cities: int, include_other_destinations: bool
) -> None:
    ingest(
        country,
        max_cities,
        include_other_destinations,
        Path(CHUNK_PATH),
        MAX_CHUNK_CHARS,
        OVERLAP_CHARS,
    )
    run(
        Path(CHUNK_PATH),
        ESL_URL,
        DEFAULT_INDEX,
        DEFAULT_EMBEDDING_MODEL,
        recreate=True,
    )


RELEVANCE_COLORS = {
    "RELEVANT": "#2e7d32",
    "PARTLY_RELEVANT": "#b8860b",
    "NON_RELEVANT": "#c62828",
    "UNKNOWN": "#666666",
}


def render_answer_meta(answer_data: dict) -> None:
    relevance = answer_data.get("relevance", "N/A")
    color = RELEVANCE_COLORS.get(relevance, "#666666")
    st.markdown(
        f"<span style='display:inline-block;background-color:{color}22;color:{color};"
        f"padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;"
        f"white-space:nowrap;'>{relevance}</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Response time", f"{answer_data.get('response_time', 0):.2f}s")
    col2.metric("Total tokens", answer_data.get("total_tokens", "N/A"))

    explanation = answer_data.get("relevance_explanation")
    if explanation:
        st.caption(f"Judge's note: {explanation}")

    cost = answer_data.get("gemini_cost")
    if cost is not None:
        st.caption(f"Estimated cost: ${cost:.6f}")


def render_feedback_buttons(idx: int, msg: dict) -> None:
    """+1/-1 for one assistant turn. Disappears once clicked so a turn can't be scored twice."""
    cid = msg.get("conversation_id")
    if not cid:
        return  # nothing to attach feedback to (e.g. the save failed for this turn)

    if msg.get("feedback") is not None:
        note = "Thanks for the feedback!" if msg["feedback"] == -1 else "Thanks!"
        st.caption(note)
        return

    col1, col2, _ = st.columns([1, 1, 8])
    if col1.button("+1", key=f"up_{idx}"):
        save_feedback(cid, "user", score=1)
        msg["feedback"] = 1
        st.rerun()
    if col2.button("-1", key=f"down_{idx}"):
        save_feedback(cid, "user", score=-1)
        msg["feedback"] = -1
        st.rerun()


# --- Sidebar: knowledge base status + build controls -----------------------

with st.sidebar:
    st.header("Knowledge base")

    init_db_and_feedback()
    engine = get_search_engine()
    count = doc_count(engine)
    if count:
        st.success(f"{count} chunks indexed")
    else:
        st.warning("Index is empty — build it below before asking questions.")

    st.divider()
    st.subheader("Build / rebuild")
    country = st.text_input("Country (Wikivoyage page title)", value="South Korea")
    max_cities = st.slider("Max cities", 1, 9, 9)
    include_other = st.checkbox(
        "Include 'other destinations'",
        value=False,
        help="Turn this on to also pick up destinations that Wikivoyage "
        "lists outside the main 'Cities' section.",
    )

    if st.button("Rebuild knowledge base", type="primary"):
        with st.spinner(
            f"Ingesting and indexing {country} — this can take a few minutes..."
        ):
            try:
                build_knowledge_base(country, max_cities, include_other)
            except Exception as e:
                st.error(f"Build failed: {e}")
            else:
                st.success("Knowledge base rebuilt.")
                st.rerun()

# --- Main chat area ----------------------------------------------------------

st.title("\U0001f9f3 Travel Assistant")
st.caption(
    "Ask about destinations, itineraries, or things to do — answers are grounded in Wikivoyage/Wikipedia content."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            with st.expander("Details"):
                render_answer_meta(msg["meta"])
            render_feedback_buttons(idx, msg)

prompt = st.chat_input("Ask a travel question...")

if prompt:
    if count == 0:
        st.error("The knowledge base is empty — build it from the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            conversation_id = None
            with st.spinner("Thinking..."):
                try:
                    answer_data = rag(engine, NATURAL_PROMPT_TEMPLATE, prompt, MODEL)
                    record = to_log_record(answer_data, prompt, NATURAL_PROMPT_TEMPLATE)
                    conversation_id = save_conversation(record, country)
                    st.session_state.conversation_id = conversation_id

                    save_feedback(
                        conversation_id, record.model, relevance=answer_data.get("relevance"), explanation=answer_data.get("relevance_explanation")
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    answer_data = None

            if answer_data:
                st.markdown(answer_data["answer"])
                with st.expander("Details"):
                    render_answer_meta(answer_data)

                new_msg = {
                    "role": "assistant",
                    "content": answer_data["answer"],
                    "meta": answer_data,
                    "conversation_id": conversation_id,
                    "feedback": None,
                }
                # Append before rendering buttons — a click triggers an
                # immediate st.rerun(), so the message must already be in
                # session_state or it would vanish on that rerun.
                st.session_state.messages.append(new_msg)
                render_feedback_buttons(len(st.session_state.messages) - 1, new_msg)