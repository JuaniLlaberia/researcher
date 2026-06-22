"""
State channel reducers. With the per-hypothesis `Send` fan-out, multiple branches
write `hypotheses` and `findings` concurrently. These reducers merge updates by id
(last-writer-wins per id) instead of letting parallel branches clobber the list.
"""
from typing import List

from src.researcher.models import Hypothesis, Finding

def merge_hypotheses(left: List[Hypothesis], right: List[Hypothesis]) -> List[Hypothesis]:
    """
    Merge incoming hypotheses into the existing list by id: an incoming hypothesis
    with a known id replaces it (e.g. a Critic status update), a new id is appended.

    Args:
        left (List[Hypothesis]): Current hypotheses in state.
        right (List[Hypothesis]): Incoming update from a node/branch.
    Returns:
        List[Hypothesis]: Merged list, original order preserved, new ids appended.
    """
    by_id = {h.id: h for h in left}
    order = [h.id for h in left]
    for h in right:
        if h.id not in by_id:
            order.append(h.id)
        by_id[h.id] = h
    return [by_id[i] for i in order]

def merge_findings(left: List[Finding], right: List[Finding]) -> List[Finding]:
    """
    Merge incoming findings into the existing list by id, so concurrent Literature
    Reviewer branches accumulate rather than overwrite, and a resumed session's
    loaded findings are not duplicated by re-runs.

    Args:
        left (List[Finding]): Current findings in state.
        right (List[Finding]): Incoming update from a node/branch.
    Returns:
        List[Finding]: Merged list, original order preserved, new ids appended.
    """
    by_id = {f.id: f for f in left}
    order = [f.id for f in left]
    for f in right:
        if f.id not in by_id:
            order.append(f.id)
        by_id[f.id] = f
    return [by_id[i] for i in order]
