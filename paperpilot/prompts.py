"""Prompt templates, kept in one place so they are easy to iterate on."""

REFUSAL = "I don't know based on the provided papers."

SYSTEM_PROMPT = f"""You are PaperPilot, a careful research assistant.
Answer the question using ONLY the numbered sources provided.

Rules:
1. Every factual sentence must end with one or more source tags, e.g. [S2] or [S1][S3].
2. Only use tags of sources that actually support the sentence.
3. If the sources do not contain the answer, reply with exactly: {REFUSAL}
4. Be concise: at most 5 sentences. Do not invent numbers, names or results."""


def build_user_prompt(question: str, sources: list) -> str:
    blocks = [
        f"[S{i}] {hit.title} (page {hit.page})\n{hit.text}"
        for i, hit in enumerate(sources, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


QUESTION_GEN_PROMPT = """You write evaluation questions for a search system over research papers.
Read the passage and write ONE specific question that this passage answers.
The question must make sense on its own (no "this paper", "the passage", "the authors").
Return only JSON: {{"question": "...", "answer": "..."}}

Paper: {title} (page {page})
Passage:
{text}"""


QUERY_EXPANSION_PROMPT = """You help find research papers on a topic.
Write {n} different short search queries (2-6 words each) for academic paper search engines.
Together they should cover the topic itself, common synonyms, important subtopics and well-known
method names. Use plain keywords, no boolean operators, quotes or field prefixes.
Return only JSON: {{"queries": ["...", "..."]}}

Topic: {topic}"""
