import os
from dotenv import load_dotenv
from src.assistant import DocumentAssistant

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables")
    exit(1)

def main():
    assistant = DocumentAssistant()

    sample_doc = """
    Q3 Financial Report: Total revenue was $450,000, up from $380,000 in Q2.
    Operating expenses were $210,000. Net profit margin improved due to reduced
    marketing spend. The healthcare division reported 1,200 new patient visits.
    """

    examples = [
        "What was the total revenue in Q3?",
        "Summarize this report in two sentences.",
        "Calculate the increase from Q2 to Q3 revenue.",
    ]

    session_id = None
    for question in examples:
        response, session_id = assistant.ask(question, sample_doc, session_id=session_id)
        print(f"\nQ: {question}")
        print(f"A: {response.answer}")
        print(f"Confidence: {response.confidence}")
        if response.tool_calls_made:
            print(f"Tools used: {response.tool_calls_made}")

if __name__ == "__main__":
    main()
