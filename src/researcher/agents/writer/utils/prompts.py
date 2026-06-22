from langchain_core.prompts import ChatPromptTemplate

LITERATURE_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are a scientific literature analyst. You are given a research goal and a set of findings (factual claims extracted from papers, each with a stance toward the research and a source paper). Synthesize them into a coherent analysis.

Work strictly from the provided findings — do not introduce claims, results, or sources that are not present. Organize the evidence by THEME (a topic or sub-question), not paper-by-paper.

Produce:
- themes: clusters of related findings. For each, write a synthesis that ties the findings together and notes where they agree and disagree. List the source_paper_ids of the findings the theme draws on.
- consensus: points supported by multiple findings (lean on findings whose stance is 'supports').
- contradictions: points where findings conflict (lean on 'contradicts' findings and tensions between sources).
- gaps: sub-questions the research goal raises that the findings do not address.

Citations: when a statement in a theme's synthesis rests on a specific paper, cite it inline with its bracket number, e.g. [1] or [2][3], using the numbering in the Sources list below. Only cite papers that appear in Sources.

If there are no findings, return empty themes and empty lists — do not fabricate.

Research Goal:
{research_goal}

Findings (each with content, stance, and source_paper_id):
{findings}

Sources (numbered; cite with these [n] markers):
{sources_block}
""")

GENERATE_REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are writing the final report for a research session. You are given the research goal, the hypotheses that were investigated, and a structured literature analysis. Produce a clear, well-grounded report.

Work strictly from the provided material — do not invent findings, outcomes, or sources.

Produce:
- title: a concise, informative title for the report.
- introduction: state the research goal and what was explored.
- hypotheses_assessment: one entry per hypothesis. Use each hypothesis's status, confidence, and its evidence_for / evidence_against to explain where it stands and why. Do not overstate: reflect the actual status and confidence.
- key_findings: a narrative of what was found, drawn from the literature analysis (its themes, consensus, and contradictions).
- open_questions: what remains unresolved, drawn from the analysis's gaps.
- conclusion: a short closing synthesis of the state of the research.

Citations: when a statement in key_findings rests on a specific paper, cite it inline with its bracket number, e.g. [1] or [2][3], using the numbering in the Sources list below. Only cite papers that appear in Sources. Do not produce a references list — it is generated separately.

Research Goal:
{research_goal}

Hypotheses (each with text, status, confidence, evidence_for, evidence_against):
{hypotheses}

Literature Analysis (themes, consensus, contradictions, gaps):
{literature_review}

Sources (numbered; cite with these [n] markers):
{sources_block}
""")

GENERATE_SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You write a short executive abstract of a completed research report. You are given the research goal and the finished report.

Write 4 to 8 sentences covering: the goal, where the hypotheses landed, the headline findings, and the key open questions. This is purely an abstraction of the report — do not introduce any new claims, and do not add citations.

Research Goal:
{research_goal}

Report:
{report}
""")
