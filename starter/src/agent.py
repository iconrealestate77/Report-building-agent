import os
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI

from .schemas import UserIntent, AnswerResponse
from .prompts import INTENT_CLASSIFICATION_PROMPT, get_chat_prompt_template
from .tools import calculator

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o"),
    temperature=float(os.getenv("TEMPERATURE", "0.1")),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1",
)

structured_intent_llm = llm.with_structured_output(UserIntent)
structured_answer_llm = llm.with_structured_output(AnswerResponse)

class AgentState(TypedDict):
    user_input: str
    document_context: str
    intent: Optional[UserIntent]
    tool_calls_made: List[str]
    final_response: Optional[AnswerResponse]

def classify_intent_node(state: AgentState) -> AgentState:
    prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=state["user_input"])
    intent = structured_intent_llm.invoke(prompt)
    state["intent"] = intent
    return state

def qa_node(state: AgentState) -> AgentState:
    template = get_chat_prompt_template("qa")
    messages = template.format_messages(
        document_context=state["document_context"], user_input=state["user_input"]
    )
    response = structured_answer_llm.invoke(messages)
    response.sources = ["document_context"]
    state["final_response"] = response
    return state

def summarize_node(state: AgentState) -> AgentState:
    template = get_chat_prompt_template("summarize")
    messages = template.format_messages(
        document_context=state["document_context"], user_input=state["user_input"]
    )
    response = structured_answer_llm.invoke(messages)
    response.sources = ["document_context"]
    state["final_response"] = response
    return state

def calculate_node(state: AgentState) -> AgentState:
    template = get_chat_prompt_template("calculate")
    llm_with_tools = llm.bind_tools([calculator])
    messages = template.format_messages(
        document_context=state["document_context"], user_input=state["user_input"]
    )
    ai_msg = llm_with_tools.invoke(messages)

    tool_calls_made = []
    tool_results = []
    for tc in getattr(ai_msg, "tool_calls", []):
        result = calculator.invoke(tc["args"])
        tool_calls_made.append(f"calculator({tc['args'].get('expression')}) = {result}")
        tool_results.append(result)

    answer_text = ai_msg.content if ai_msg.content else "; ".join(tool_results)
    state["final_response"] = AnswerResponse(
        answer=answer_text or "No calculation could be performed.",
        confidence=0.95 if tool_results else 0.4,
        sources=["document_context"],
        tool_calls_made=tool_calls_made,
    )
    return state

def route_by_intent(state: AgentState) -> str:
    return state["intent"].intent_type

def create_workflow():
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("calculate", calculate_node)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"qa": "qa", "summarize": "summarize", "calculate": "calculate"},
    )
    workflow.add_edge("qa", END)
    workflow.add_edge("summarize", END)
    workflow.add_edge("calculate", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
