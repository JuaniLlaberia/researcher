from typing import List
from pydantic import BaseModel, Field

class GeneratedHypothesis(BaseModel):
    text: str = Field(..., description="A single, self-contained candidate hypothesis: one precise, testable claim that addresses the research goal. Not a question or a topic.")

class GenerationOutput(BaseModel):
    hypotheses: List[GeneratedHypothesis] = Field(..., description="The batch of distinct candidate hypotheses generated for the research goal.")

class HypothesisScore(BaseModel):
    index: int = Field(..., description="The index of the hypothesis being scored, matching the `index` it was given in the input list.")
    relevance: float = Field(..., ge=0, le=1, description="Does the hypothesis actually address the research goal? 1 = directly on-target, 0 = unrelated.")
    testability: float = Field(..., ge=0, le=1, description="Can it be concretely tested or refuted by an experiment? 1 = clear falsifiable prediction, 0 = unfalsifiable.")
    specificity: float = Field(..., ge=0, le=1, description="Does it name a concrete mechanism, variable, or direction of effect? 1 = precise, 0 = vague/hand-wavy.")
    plausibility: float = Field(..., ge=0, le=1, description="Is it consistent with known science and the provided evidence? 1 = well-grounded, 0 = implausible/absurd.")
    novelty: float = Field(..., ge=0, le=1, description="Is it non-trivial rather than already-obvious or well-established? 1 = novel angle, 0 = trivially known.")
    rationale: str = Field(..., description="One short sentence justifying the scores.")

class ScoringOutput(BaseModel):
    scores: List[HypothesisScore] = Field(..., description="One score entry per input hypothesis, referenced by its index.")

class ScoredHypothesis(BaseModel):
    text: str = Field(..., description="The hypothesis text.")
    score: float = Field(..., ge=0, le=1, description="Weighted composite of the five dimension scores.")
    relevance: float = Field(..., ge=0, le=1)
    testability: float = Field(..., ge=0, le=1)
    specificity: float = Field(..., ge=0, le=1)
    plausibility: float = Field(..., ge=0, le=1)
    novelty: float = Field(..., ge=0, le=1)
    rationale: str = Field(..., description="One short sentence justifying the scores.")

class RefineOutput(BaseModel):
    text: str = Field(..., description="The refined hypothesis: a single, self-contained, testable claim that addresses the critique while staying true to the original intent.")
    relevance: float = Field(..., ge=0, le=1, description="Does the refined hypothesis address the research goal? 1 = directly on-target, 0 = unrelated.")
    testability: float = Field(..., ge=0, le=1, description="Can it be concretely tested or refuted? 1 = clear falsifiable prediction, 0 = unfalsifiable.")
    specificity: float = Field(..., ge=0, le=1, description="Does it name a concrete mechanism, variable, or direction of effect? 1 = precise, 0 = vague.")
    plausibility: float = Field(..., ge=0, le=1, description="Is it consistent with known science and the evidence? 1 = well-grounded, 0 = implausible.")
    novelty: float = Field(..., ge=0, le=1, description="Is it non-trivial rather than already-obvious? 1 = novel angle, 0 = trivially known.")
    rationale: str = Field(..., description="One short sentence on what the refinement changed and why.")
