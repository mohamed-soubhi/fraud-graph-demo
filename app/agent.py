"""Option B — LangGraph self-healing Cypher agent.

Graph nodes:
  generate_cypher  → LLM produces Cypher from question
  execute_cypher   → runs Cypher against Neo4j
  fix_cypher       → LLM repairs bad Cypher using error + original query
  interpret_results → LLM explains raw rows in plain English

Edges:
  generate_cypher → execute_cypher
  execute_cypher  → fix_cypher      (if error and retries <= MAX_RETRIES)
  execute_cypher  → interpret_results (if success)
  execute_cypher  → END              (if error and retries > MAX_RETRIES)
  fix_cypher      → execute_cypher
  interpret_results → END
"""
import time
from typing import Optional
from typing_extensions import TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from config import CFG

MAX_RETRIES: int = CFG["agent_max_retries"]


# ── state ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:       str
    entities:       str
    cypher:         Optional[str]
    cypher_error:   Optional[str]
    rows:           Optional[list]
    interpretation: Optional[str]
    retries:        int
    cypher_ms:      float
    fix_ms:         float
    query_ms:       float
    interpret_ms:   float


# ── prompts ───────────────────────────────────────────────────────────────────

FIX_PROMPT = PromptTemplate.from_template("""
You are a Neo4j Cypher expert. The query below failed with an error.
Fix the Cypher query so it runs correctly against the given schema.
Return ONLY the corrected Cypher — no markdown, no explanation.

Schema:
{schema}

Original question: {question}

Failed Cypher:
{cypher}

Error:
{error}

Fixed Cypher:
""")


# ── graph factory ─────────────────────────────────────────────────────────────

def build_agent(llm, driver, cypher_chain, interpret_chain, schema: str):
    """Return a compiled LangGraph agent.

    Parameters
    ----------
    llm            : OllamaCloudLLM instance
    driver         : Neo4j GraphDatabase driver
    cypher_chain   : PROMPT | llm  (from chat.py)
    interpret_chain: INTERPRET_PROMPT | llm  (from chat.py)
    schema         : SCHEMA string (from chat.py)
    """
    fix_chain = FIX_PROMPT | llm

    # ── nodes ─────────────────────────────────────────────────────────────────

    def generate_cypher(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        try:
            cypher = cypher_chain.invoke({
                "schema":   schema,
                "entities": state["entities"],
                "question": state["question"],
            }).strip()
        except Exception as e:
            return {**state,
                    "cypher": None,
                    "cypher_error": f"LLM error during generation: {e}",
                    "retries": MAX_RETRIES + 1,   # skip straight to END — can't fix without a query
                    "cypher_ms": state["cypher_ms"] + (time.perf_counter() - t0) * 1000}
        return {**state,
                "cypher": cypher,
                "cypher_error": None,
                "cypher_ms": state["cypher_ms"] + (time.perf_counter() - t0) * 1000}

    def execute_cypher(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        try:
            with driver.session() as session:
                rows = session.run(state["cypher"]).data()
            return {**state,
                    "rows": rows,
                    "cypher_error": None,
                    "query_ms": state["query_ms"] + (time.perf_counter() - t0) * 1000}
        except Exception as e:
            return {**state,
                    "rows": None,
                    "cypher_error": str(e),
                    "retries": state["retries"] + 1,
                    "query_ms": state["query_ms"] + (time.perf_counter() - t0) * 1000}

    def fix_cypher(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        fixed = fix_chain.invoke({
            "schema":   schema,
            "question": state["question"],
            "cypher":   state["cypher"],
            "error":    state["cypher_error"],
        }).strip()
        return {**state,
                "cypher": fixed,
                "cypher_error": None,
                "fix_ms": state["fix_ms"] + (time.perf_counter() - t0) * 1000}

    def interpret_results(state: AgentState) -> AgentState:
        if not state.get("rows"):
            return state
        t0 = time.perf_counter()
        results_str = "\n".join(str(r) for r in state["rows"][:10])
        interpretation = interpret_chain.invoke({
            "question": state["question"],
            "results":  results_str,
        }).strip()
        return {**state,
                "interpretation": interpretation,
                "interpret_ms": state["interpret_ms"] + (time.perf_counter() - t0) * 1000}

    # ── routing ───────────────────────────────────────────────────────────────

    def route_after_execute(state: AgentState) -> str:
        if state.get("cypher_error"):
            if state["retries"] <= MAX_RETRIES:
                return "fix_cypher"
            return END
        return "interpret_results"

    # ── graph ─────────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("generate_cypher",   generate_cypher)
    graph.add_node("execute_cypher",    execute_cypher)
    graph.add_node("fix_cypher",        fix_cypher)
    graph.add_node("interpret_results", interpret_results)

    graph.set_entry_point("generate_cypher")
    graph.add_edge("generate_cypher", "execute_cypher")
    graph.add_conditional_edges("execute_cypher", route_after_execute)
    graph.add_edge("fix_cypher",        "execute_cypher")
    graph.add_edge("interpret_results", END)

    return graph.compile()


def initial_state(question: str, entities: str) -> AgentState:
    return AgentState(
        question=question,
        entities=entities,
        cypher=None,
        cypher_error=None,
        rows=None,
        interpretation=None,
        retries=0,
        cypher_ms=0.0,
        fix_ms=0.0,
        query_ms=0.0,
        interpret_ms=0.0,
    )
