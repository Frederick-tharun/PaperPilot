"""Step 5b: a simple web page on top of the API.

Run:  streamlit run app/streamlit_app.py
(the API must be running; set PP_API_URL if it is not on localhost:8000)

Two tabs:
* Ask               - question -> cited answer, optionally limited to one topic
* Research a topic  - type a topic; the API finds, downloads and indexes papers
"""

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("PP_API_URL", "http://localhost:8000")
ALL_TOPICS = "All topics"
STAGES = {"expand": "Expanding the topic into search queries",
          "search": "Searching arXiv, Semantic Scholar and OpenAlex",
          "rank": "Removing duplicates and ranking by relevance",
          "download": "Downloading open-access PDFs",
          "index": "Indexing the new papers",
          "done": "Finished"}

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="centered")
st.title("📄 PaperPilot")
st.caption("Ask questions about research papers. Every answer cites paper and page.")


def load_topics() -> list[dict]:
    try:
        return requests.get(f"{API_URL}/topics", timeout=30).json()
    except Exception:
        return []


def paper_link(item: dict) -> str:
    url = item.get("url") or f"https://arxiv.org/abs/{item['paper_id']}"
    return f"[{item['paper_id']}]({url})"


with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Sources to retrieve (top-k)", 1, 10, 5)
    topics = load_topics()
    labels = [ALL_TOPICS] + [t["topic"] for t in topics]
    default = st.session_state.get("last_topic")
    index = labels.index(default) if default in labels else 0
    scope = st.selectbox("Search in", labels, index=index,
                         format_func=lambda t: t if t == ALL_TOPICS else
                         f"{t} ({next(x['chunks'] for x in topics if x['topic'] == t)} chunks)")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        st.success(f"API online · {health['indexed_chunks']} chunks · LLM: {health['llm']}")
    except Exception:
        st.error(f"API not reachable at {API_URL}")

ask_tab, research_tab = st.tabs(["💬 Ask", "🔎 Research a new topic"])

# ---------------------------------------------------------------- Ask
with ask_tab:
    question = st.text_input("Your question",
                             placeholder="How is the spectral radius of an echo state network chosen?")
    if st.button("Ask", type="primary") and question.strip():
        payload = {"question": question, "top_k": top_k, "topic": None if scope == ALL_TOPICS else scope}
        with st.spinner("Searching papers and writing an answer..."):
            try:
                response = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        if data["refused"]:
            st.warning(data["answer"])
        else:
            st.markdown(data["answer"])
            if data["citations"]:
                st.markdown("**Citations**")
                for c in data["citations"]:
                    st.markdown(f"- `[{c['tag']}]` {c['title']} — page {c['page']} ({paper_link(c)})")

        with st.expander("Retrieved sources"):
            for i, s in enumerate(data["sources"], start=1):
                st.markdown(f"**[S{i}] {s['title']} — page {s['page']}** · similarity {s['score']:.2f}")
                st.caption(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))

# ---------------------------------------------------------------- Research
with research_tab:
    st.write("Type any research topic. PaperPilot expands it into several searches, queries three "
             "paper databases, removes duplicates, ranks by relevance, downloads open-access PDFs "
             "and indexes them so you can ask questions right away.")
    topic = st.text_input("Topic", placeholder="graph neural networks for drug discovery")
    n_papers = st.slider("Number of papers", 10, 100, 40, step=5)

    if st.button("Research topic", type="primary") and topic.strip():
        try:
            job = requests.post(f"{API_URL}/research", json={"topic": topic, "max_papers": n_papers},
                                timeout=30).json()
        except Exception as exc:
            st.error(f"Could not start research: {exc}")
            st.stop()

        with st.status(f"Researching “{topic}” …", expanded=True) as status:
            stage_box = st.empty()
            bar = st.progress(0.0)
            log_box = st.empty()
            while True:
                time.sleep(1.5)
                try:
                    job = requests.get(f"{API_URL}/research/{job['job_id']}", timeout=30).json()
                except Exception as exc:
                    st.error(f"Lost contact with the API: {exc}")
                    st.stop()
                stage_box.markdown(f"**{STAGES.get(job['stage'], 'Starting…')}**  \n{job['message']}")
                if job.get("total"):
                    bar.progress(min(job["done"] / job["total"], 1.0))
                log_box.code("\n".join(job["log"][-12:]) or "…")
                if job["status"] in {"done", "failed"}:
                    break
            if job["status"] == "failed":
                status.update(label="Research failed", state="error")
                st.error(job["error"])
                st.stop()
            status.update(label=f"Done: “{topic}”", state="complete", expanded=False)

        r = job["result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates found", r["candidates"])
        c2.metric("Papers downloaded", r["downloaded"])
        c3.metric("Chunks indexed", r["chunks_indexed"])
        st.caption("Queries: " + " · ".join(r["queries"]) +
                   f"  \nDuplicates removed: {r['duplicates_removed']} · without PDF: {r['without_pdf']} · "
                   f"already in library: {r['already_known']} · report: `{r['report_path']}`")
        with st.expander("Papers", expanded=True):
            for p in r["papers"]:
                mark = "✅" if p["downloaded"] else "⚠️"
                st.markdown(f"{mark} {p['score']:.2f} · [{p['title']}]({p['url']})")
        if r["downloaded"]:
            st.session_state["last_topic"] = r["topic"]
            st.success("Switch to the **Ask** tab — the sidebar is now set to this topic.")
            st.button("Refresh topics")  # rerun so the sidebar dropdown picks up the new topic
