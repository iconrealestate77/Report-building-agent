from langchain_core.prompts import ChatPromptTemplate

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a document assistant.
Classify the user's request into exactly one of these categories:

- "qa": The user is asking a specific factual question about document content
  (e.g. "What was the total revenue in Q3?", "Who is the patient's primary physician?")
- "summarize": The user wants an overview, summary, or key points extracted
  (e.g. "Summarize this report", "What are the main takeaways?")
- "calculate": The user wants a mathematical operation performed on numbers in the document
  (e.g. "What's the sum of all expenses?", "Calculate the percentage increase")

Respond with the intent_type, a confidence score (0-1) reflecting how certain you are,
and a brief one-sentence reasoning for your classification.

User request: {user_input}
"""

def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """Returns the appropriate system+human prompt template based on classified intent."""
    system_messages = {
        "qa": (
            "You are a precise document Q&A assistant. Answer only using information "
            "found in the provided document context. If the answer isn't in the document, "
            "say so clearly rather than guessing. Cite the relevant section when possible."
        ),
        "summarize": (
            "You are a concise summarization assistant. Produce a clear, well-organized "
            "summary of the document, highlighting key points, figures, and conclusions. "
            "Keep it shorter than the original while preserving essential meaning."
        ),
        "calculate": (
            "You are a careful calculation assistant. Identify the relevant numbers in the "
            "document, use the calculator tool to compute results precisely, and show your "
            "reasoning. Never guess at arithmetic — always use the tool."
        ),
    }
    system_msg = system_messages.get(intent_type, system_messages["qa"])
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Document context:\n{document_context}\n\nUser request: {user_input}"),
    ])
