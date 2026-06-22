from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You maintain a rolling summary of an ongoing research-assistant session. Your job is to fold the most recent conversation turns into the existing summary, producing a single updated summary that lets the assistant resume the session later without re-reading the full message history.

Produce an updated summary that follows these rules:
- Integrate the new messages into the prior summary as one coherent whole. Update, merge, and rewrite — do not simply append the new content to the old summary.
- Preserve durable, research-relevant facts: the research goal, hypotheses raised or revised and how their direction changed, key findings or papers referenced, decisions taken, constraints or preferences the user stated, and any open questions or agreed next steps.
- Drop greetings, acknowledgements, tool-call mechanics, and redundant back-and-forth that carry no lasting information.
- Stay strictly grounded in what was actually said. Do not invent details, results, or conclusions that do not appear in the prior summary or the new messages.
- Be concise. Write neutral, third-person notes — not a turn-by-turn transcript and not a chat reply.

If the prior summary is empty, write the summary from the new messages alone.

Prior summary (may be empty if this is the first summary of the session):
{crr_summary}
                                                  
New messages since prior summary:
{messages}

Return only the updated summary text.
""")
