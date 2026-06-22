from typing import List

from src.researcher.models import Hypothesis, Finding, HypothesisStatus
from src.researcher.agents.hypothesizer.utils.models import ScoredHypothesis
from src.researcher.agents.critic.utils.models import CriticAssessment

VERDICT_TO_STATUS: dict[str, HypothesisStatus] = {
    "holds": "active",
    "refuted": "refuted",
    "needs_refinement": "active",
    "needs_more_evidence": "active",
}

def scored_to_hypothesis(scored: ScoredHypothesis, parent_id: str | None = None) -> Hypothesis:
    """
    Promote a Hypothesizer ScoredHypothesis into a canonical Hypothesis. The
    composite score becomes the confidence; the per-dimension breakdown and
    rationale are provenance and belong in the decision trail, not the model.

    Args:
        scored (ScoredHypothesis): A scored hypothesis from generation or refinement.
        parent_id (str | None): Lineage link — set for a refined child, None for a root.
    Returns:
        Hypothesis: A fresh domain hypothesis (new id, status "active").
    """
    return Hypothesis(text=scored.text, confidence=scored.score, status="active", parent_id=parent_id)

def _condense_feedback(assessment: CriticAssessment) -> str:
    """Flatten a critique into a single feedback string for the refine step."""
    parts: List[str] = []
    if assessment.contradictions:
        parts.append("Contradictions: " + "; ".join(assessment.contradictions))
    if assessment.assumptions:
        parts.append("Unstated assumptions: " + "; ".join(assessment.assumptions))
    parts.append("Verdict rationale: " + assessment.rationale)
    return "\n".join(parts)

def apply_critic(hypothesis: Hypothesis, assessment: CriticAssessment) -> Hypothesis:
    """
    Fold a Critic assessment into a hypothesis in place: set the lifecycle status
    from the verdict, record the falsifiability score, and stash the verdict +
    condensed feedback transiently so the critic-join can route and the refine
    step has the critique. Confidence is left as the Hypothesizer composite —
    adversarial reweighting is a Stage 5 concern.

    Args:
        hypothesis (Hypothesis): The hypothesis that was assessed.
        assessment (CriticAssessment): The Critic's verdict and analysis.
    Returns:
        Hypothesis: The same hypothesis, updated.
    """
    hypothesis.status = VERDICT_TO_STATUS.get(assessment.verdict, hypothesis.status)
    hypothesis.falsifiability_score = assessment.falsifiability_score
    hypothesis.pending_verdict = assessment.verdict
    hypothesis.pending_feedback = _condense_feedback(assessment)
    return hypothesis

def park_parent(hypothesis: Hypothesis) -> Hypothesis:
    """
    Park a hypothesis that has been superseded by a refined child: mark it parked
    and clear its pending verdict so the critic-join never re-selects it for
    refinement. Kept in state for lineage, not pursued further.

    Args:
        hypothesis (Hypothesis): The parent hypothesis being refined.
    Returns:
        Hypothesis: The same hypothesis, parked.
    """
    hypothesis.status = "parked"
    hypothesis.pending_verdict = None
    hypothesis.pending_feedback = None
    return hypothesis

def stamp_findings(findings: List[Finding], hypothesis_id: str) -> List[Finding]:
    """
    Attach the owning hypothesis id to a batch of findings returned by the
    Literature Reviewer (which has no notion of hypothesis id), so they can be
    accumulated in state and persisted against the right hypothesis.

    Args:
        findings (List[Finding]): Findings for a single hypothesis.
        hypothesis_id (str): The hypothesis these findings were gathered for.
    Returns:
        List[Finding]: The same findings, each stamped with hypothesis_id.
    """
    for finding in findings:
        finding.hypothesis_id = hypothesis_id
    return findings
