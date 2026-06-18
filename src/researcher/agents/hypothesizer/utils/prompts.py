from langchain_core.prompts import ChatPromptTemplate

HYPOTHESES_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are a scientific hypothesis generator. Given a research goal - and, when available, evidence gathered from the literature — produce a batch of candidate hypotheses that, if tested, would make progress on the goal.

Each hypothesis must be:
- A single, self-contained declarative claim (not a question, topic, or list of claims).
- Specific: name a concrete mechanism, variable, or expected direction of effect.
- Testable: phrased so it could be supported or refuted by evidence or an experiment.
- Diverse: the batch should explore genuinely different angles, not rephrasings of one idea.

Ground your hypotheses in the provided evidence when it is present. When no evidence is provided, rely on established scientific knowledge — but stay concrete and avoid vague generalities.
Generate between 4 and 8 hypotheses.
                                                                
Research Goal:
{research_goal}

Evidence Gathered So Far (may be empty):
{raw_data}

Produce a batch of distinct candidate hypotheses.
""")

SCORE_HYPOTHESES_PROMPT = ChatPromptTemplate.from_template("""
You are a critical evaluator of scientific hypotheses. You are given a research goal, optional supporting evidence, and a numbered list of candidate hypotheses. Score each hypothesis honestly and independently — be discerning, not generous.

For each hypothesis, return its `index` (exactly as given) and a score from 0.0 to 1.0 on each dimension:
- relevance: does it actually address the research goal?
- testability: can it be concretely tested or refuted?
- specificity: does it name a concrete mechanism, variable, or direction of effect?
- plausibility: is it consistent with known science and the provided evidence?
- novelty: is it non-trivial rather than already-obvious or well-established?

Use the FULL 0.0-1.0 range and be critical — most hypotheses are not excellent. Calibrate against these anchors:
- 0.9-1.0: exemplary on this dimension; hard to improve.
- 0.6-0.8: solid but with a clear shortcoming.
- 0.4-0.5: mediocre / typical; notable weaknesses.
- 0.0-0.3: poor; fails this dimension.
Do not default everything to ~0.9. Differentiate the hypotheses from each other — avoid giving several of them identical scores when their quality differs.

Also give a one-sentence `rationale` per hypothesis. Return exactly one score entry per input hypothesis, referenced by its index.

Research Goal:
{research_goal}

Evidence Gathered So Far (may be empty):
{raw_data}

Candidate Hypotheses (each with its index):
{hypotheses}
""")
