from langchain_core.prompts import ChatPromptTemplate

ADVERSARIAL_QUERIES_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You generate search queries for the adversarial step of a literature review. Your goal is to find papers that would CONTRADICT, weaken, or expose flaws in a given hypothesis — not papers that support it.

Produce 1 to 3 concise, information-dense queries in precise scientific terms (key methods, concepts, entities). 
Approach the hypothesis from angles likely to surface counter-evidence: competing methods that outperform it, known limitations or failure cases, negative or null results, and critiques of its underlying assumptions.

Research Goal:
{research_goal}

Hypothesis Under Scrutiny:
{hypothesis}

Produce the adversarial queries.
""")

ADVERSARIAL_RELEVANCE_GATE_PROMPT = ChatPromptTemplate.from_template("""
You decide which candidate papers are worth the cost of full ingestion for the adversarial review of a hypothesis. You are given each candidate's title and abstract only.

Select the papers whose title and abstract suggest they likely contain evidence AGAINST the hypothesis: contradicting findings, competing approaches that do better, limitations, negative results, or challenges to its assumptions. 
Be selective: ingestion is expensive, so favor precision and skip papers that merely support the hypothesis or are only loosely related. Selecting none is acceptable.

Research Goal:
{research_goal}

Hypothesis Under Scrutiny:
{hypothesis}

Candidate Papers (each with its index, title, and abstract):
{candidates}

Return selected_indices: the list of `index` values of the papers you chose.
""")

HYPOTHESIS_ASSESSMENT_PROMPT = ChatPromptTemplate.from_template("""
You are a rigorous adversarial critic of scientific hypotheses. Given a hypothesis and evidence gathered to challenge it, produce a structured assessment and a verdict. Your job is to stress-test the hypothesis: take the contradicting evidence seriously and do not give the hypothesis the benefit of the doubt.

Produce:
- contradictions: specific points where the evidence contradicts or undermines the hypothesis. If the evidence contains no genuine contradictions, return an empty list — do not invent them.
- assumptions: hidden or unstated assumptions the hypothesis relies on that may not hold.
- falsifiability_score: 0.0-1.0, how concretely testable/refutable the hypothesis is (1 = clear falsifiable prediction, 0 = unfalsifiable).
- rationale: one or two sentences justifying the verdict.
- verdict, exactly one of:
  - "holds": withstands the adversarial evidence; no serious contradictions or fatal assumptions.
  - "refuted": the contradicting evidence clearly outweighs any support.
  - "needs_refinement": promising but has a specific flaw or shaky assumption that should be fixed.
  - "needs_more_evidence": the gathered evidence is too thin or off-target to judge.

If no contradicting evidence was found, that is itself a signal the hypothesis may hold — do not refute a hypothesis for lack of evidence; use "needs_more_evidence" when the search came up empty or irrelevant.

Research Goal:
{research_goal}

Hypothesis Under Scrutiny:
{hypothesis}

Evidence Gathered to Challenge It (may be empty):
{raw_data}
""")
