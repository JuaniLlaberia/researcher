from langchain_core.prompts import ChatPromptTemplate

QUERIES_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You generate search queries for a scientific literature review. Your queries will be used to retrieve papers that help evaluate a specific hypothesis within a broader research goal.

Produce:
- main_query: one concise, information-dense query capturing the core of what must be found to test the hypothesis. Write it in precise scientific terms (key methods, concepts, entities), not as a full sentence or question.
- queries: 1 to 3 diverse variants of the main query that approach it from different angles — synonyms, broader or narrower scope, alternative terminology, or related sub-topics — to widen retrieval coverage. Avoid near-duplicates of the main query.

Keep every query tightly focused on the hypothesis and goal. If feedback is provided, address it directly (e.g. broaden, narrow, or shift the topic as instructed).

Research Goal:
{research_goal}

Hypothesis Under Investigation:
{hypothesis}

Feedback From a Previous Attempt (may be empty on the first try):
{feedback}
""")

RELEVANCE_GATE_PROMPT = ChatPromptTemplate.from_template("""
You decide which candidate papers are worth the cost of full ingestion (downloading and processing) for a literature review. You are given each candidate's title and abstract only.

Select the papers whose title and abstract indicate they likely contain evidence bearing on the hypothesis: supporting it, contradicting it, or providing directly relevant context. Be selective: ingestion is expensive, so favor precision over recall and skip papers that are only loosely related, off-topic, or redundant. Selecting none is acceptable.

Research Goal:
{research_goal}

Hypothesis Under Investigation:
{hypothesis}

Candidate Papers (each with its index, title, and abstract):
{candidates}

Return selected_indices: the list of `index` values of the papers you chose.
""")

DATA_VALIDATION_PROMPT = ChatPromptTemplate.from_template("""
You assess whether the evidence gathered so far is enough to draw meaningful findings about a hypothesis, or whether the search should continue.

Judge the evidence and assign one label:
- "sufficient": the evidence covers the hypothesis well enough to extract grounded findings (including contradicting evidence where it exists).
- "insufficient": the evidence is on-topic but too sparse, shallow, or one-sided to draw reliable findings — more of the same kind of evidence is needed.
- "needs_different_queries": the evidence does not actually address the hypothesis — the search is pointed in the wrong direction and a different angle is needed.

Also provide feedback: concrete guidance for the query generator on what is missing or what to try next. If the label is "sufficient", feedback may be an empty string.

Research Goal:
{research_goal}

Hypothesis Under Investigation:
{hypothesis}

Evidence Gathered So Far:
{raw_data}
""")

FINDING_EXTRATOR_PROMPT = ChatPromptTemplate.from_template("""
You extract structured findings from gathered literature for a research review. A finding is a single, self-contained factual claim that is grounded in the evidence and relevant to the hypothesis.

For each distinct claim in the evidence that bears on the hypothesis, produce a finding with:
- content: the claim, stated in one or two precise sentences. Use only what the evidence actually says: do not infer, extrapolate, or add outside knowledge.
- stance: "supports" if the claim is evidence for the hypothesis, "contradicts" if it is evidence against it, or "neutral" if it is relevant context but neither.
- source_paper_id: the paper_id of the evidence passage the claim came from.

Extract only claims clearly supported by the evidence text. Do not invent findings. If the evidence contains no relevant claims, return an empty list.

Research Goal:
{research_goal}

Hypothesis Under Investigation:
{hypothesis}

Evidence Gathered So Far:
{raw_data}
""")
