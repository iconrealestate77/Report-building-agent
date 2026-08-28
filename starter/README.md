# Document Assistant

A multi-agent document processing system built with LangChain and LangGraph. The assistant classifies each user request into one of three intents — question answering, summarization, or calculation — and routes it to a specialized node that handles that task, returning a validated, structured response.

## Getting Started

### Dependencies

```
langchain
langchain-openai
langgraph
pydantic
python-dotenv
```

### Installation

1. Clone the repository and navigate into the `starter/` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API key:
   ```bash
   cp .env.example .env
   ```
   ```
   OPENAI_API_KEY=your_key_here
   MODEL_NAME=gpt-4o
   TEMPERATURE=0.1
   SESSION_STORAGE_PATH=./sessions
   ```
4. Run the assistant:
   ```bash
   python main.py
   ```

## Project Instructions

### Architecture Overview

The system is a `StateGraph` with four nodes:

```
classify_intent → (qa | summarize | calculate) → END
```

`classify_intent` is the single entry point. It calls the LLM with a structured-output wrapper bound to the `UserIntent` schema, so the classification itself is never free-text — it's always a validated Pydantic object with an `intent_type`, a `confidence` score, and a `reasoning` string. `route_by_intent` reads `intent_type` off that object and LangGraph's conditional edges send the state to exactly one of the three task nodes. Each task node produces a validated `AnswerResponse` and the graph terminates.

### Why This Structure

I kept classification as its own node rather than folding it into the task nodes so that intent and answer generation are independently testable and independently logged — you can see *why* the router made a decision separately from *what* the answer was, which is useful both for debugging misroutes and for the confidence-scoring requirement in the rubric.

The calculation node is the one node that binds a tool (`calculator`) to the LLM rather than asking the model to produce structured output directly. Math is exactly the place where I don't trust the model to compute a value inline — the system prompt explicitly instructs it to always call the tool rather than guess at arithmetic, and the node only accepts the tool's returned string as the numeric result.

### Schemas and Validation

**`UserIntent`**
- `intent_type: Literal["qa", "summarize", "calculate"]` — restricted to exactly three values; anything else raises a Pydantic validation error at the schema level, before it ever reaches the router.
- `confidence: float` — constrained with `Field(ge=0.0, le=1.0)`. A value like `1.5` is rejected immediately with a `ValidationError`, rather than being silently clamped or accepted.
- `reasoning: str` — a short justification, mainly useful for debugging and for showing classification quality in the example conversations below.

**`AnswerResponse`**
- `answer: str` — the actual content returned to the user.
- `confidence: float`, same `0–1` constraint as above, defaulting to `0.8` when a node doesn't explicitly set it.
- `sources: List[str]` — which document sections informed the answer (currently `["document_context"]`, since the project uses a single passed-in context; this list is where citation-style sourcing would expand if retrieval were added).
- `tool_calls_made: List[str]` — a human-readable record of any tool invocations (e.g. `"calculator(450000 - 380000) = 70000"`), populated only by the calculation node.

Both schemas are enforced by calling `llm.with_structured_output(Schema)`, which forces the model's output through the schema's JSON schema at the API level — the LLM cannot return a response that fails validation; LangChain re-prompts or raises rather than passing through malformed data.

### Tool: Calculator

`calculator` is decorated with `@tool` so LangChain can expose it to the LLM as a callable function. Internally, it does **not** use Python's `eval()`. Instead, it parses the expression with `ast.parse(expression, mode="eval")` and walks the resulting AST, only permitting a fixed whitelist of numeric operators (`+ - * / ** % ` and unary negation). Anything outside that whitelist — including attempts to import modules, call arbitrary functions, or access attributes — raises `ValueError` before any code executes, and the failure is returned as a string rather than propagating an exception up to the graph. Every call, successful or not, is appended as a JSON line to `logs/tool_calls.jsonl` with a timestamp, the expression, the result, and a success flag. The tool always returns `str(result)`, never a raw `int`/`float`, per the interface the rest of the graph expects.

### Prompt Engineering

`INTENT_CLASSIFICATION_PROMPT` gives the classifier explicit category definitions with two examples per category, and asks for `intent_type`, `confidence`, and `reasoning` — matching the `UserIntent` schema field-for-field. `get_chat_prompt_template(intent_type)` returns a different `ChatPromptTemplate` for each intent: the QA prompt emphasizes staying strictly within the provided document context and admitting when an answer isn't present; the summarization prompt emphasizes brevity and coverage of key figures; the calculation prompt explicitly forbids the model from computing results itself and instructs it to rely on the calculator tool.

### State and Memory

Graph state (`AgentState`, a `TypedDict`) carries `user_input`, `document_context`, `intent`, `tool_calls_made`, and `final_response` through every node. State is threaded end-to-end: `classify_intent_node` populates `intent`; the task node the router selects reads `document_context`/`user_input` and populates `final_response`; nothing downstream needs to re-derive earlier values.

Conversation memory is handled at two levels:
- **Within a run**, LangGraph's `MemorySaver` checkpointer is attached at compile time (`workflow.compile(checkpointer=memory)`), and every invocation is scoped to a `thread_id` equal to the session ID, so a multi-turn conversation can be resumed by reusing the same session ID.
- **Across runs**, `DocumentAssistant._log_session()` appends a JSON line per turn to `sessions/{session_id}.jsonl`, containing the timestamp, user input, the full `UserIntent`, and the full `AnswerResponse`. This is the durable session history the rubric asks for, independent of whatever LangGraph keeps in memory for a live process.

### Example Conversations

The three intents were tested against this sample document:

```
Q3 Financial Report: Total revenue was $450,000, up from $380,000 in Q2.
Operating expenses were $210,000. Net profit margin improved due to reduced
marketing spend. The healthcare division reported 1,200 new patient visits.
```

**Q&A**
```
Q: What was the total revenue in Q3?
A: The total revenue in Q3 was $450,000.
confidence=0.95  sources=['document_context']  tool_calls_made=[]
```

**Summarization**
```
Q: Summarize this report in two sentences.
A: In Q3, total revenue increased to $450,000 from $380,000 in Q2, with operating
   expenses at $210,000, leading to an improved net profit margin due to reduced
   marketing spend. The healthcare division also saw growth with 1,200 new patient visits.
confidence=0.95  sources=['document_context']  tool_calls_made=[]
```

**Calculation**
```
Q: Calculate the increase from Q2 to Q3 revenue.
A: 70000
confidence=0.95  sources=['document_context']
tool_calls_made=['calculator(450000 - 380000) = 70000']
```

In the calculation example, `tool_calls_made` shows the exact expression the model chose and the tool's returned value — confirming the arithmetic was performed by `calculator()` rather than generated by the LLM directly.

## Built With

* [LangChain](https://www.langchain.com/) - LLM application framework; structured output, prompt templates, and the `@tool` decorator
* [LangGraph](https://www.langchain-ai.github.io/langgraph/) - Stateful graph orchestration, conditional routing, and checkpointed memory
* [Pydantic](https://docs.pydantic.dev/) - Schema definition and runtime validation for `UserIntent` and `AnswerResponse`
* [OpenAI API](https://platform.openai.com/) (via Vocareum proxy) - Underlying LLM for classification and generation
