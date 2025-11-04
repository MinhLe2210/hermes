import os
import sys
import json
import datetime
import dotenv
import streamlit as st

from src.prompts.prompt import *
from src.logger.logger import get_logger
from json_repair import repair_json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from src.semantic_cache.operations import get_from_cache, set_in_cache
from src.utils.helpers import extract_code_block, run_generated_code
from src.prompts.prompt import *
from src.llm.llm_client import LLM
from src.utils.predict_lr import predict_model

logger = get_logger(__name__)

class State(TypedDict):
    query: str
    from_cache: bool
    analysis_type: Literal["data", "plot", "predict"] | None
    result: str | None
    critique: str | None
    max_tries: int


def semantic_cache(state: State):
    with st.status("🔍 Checking semantic cache..."):
        cached_response = get_from_cache(state["query"])
        if cached_response:
            state["from_cache"] = True
            st.write("✅ Cache Hit:", cached)
        else:
            st.write("🚫 Cache Miss — calling LLM...")
            state["from_cache"] = False
    return state


def analys(state: State):
    with st.status("🧩 Analyzing query type..."):
        prompt = INTENT_CLASSIFICATION_PROMPT.replace(
            "{question}", state["query"]
        )
        try:
            llm = LLM(model_name="gemini-2.5-flash")
            res = llm.invoke(prompt)

            res_json = json.loads(repair_json(res))
            intent = res_json.get("intent")
            state["analysis_type"] = intent

        except Exception as e:
            state["max_tries"] += 1
            state["critique"] = "loop"
            logger.info(f"Error occurred: {e}")
    return state


def data_scientist_agent(state: State):
    with st.status("📊 Running data_scientist_agent..."):
        date_time = datetime.datetime.now().isoformat()
        prompt = DATAFRAME_ANALYSIS.format(
            question=state["query"],
            date_time=date_time
        )
        try:
            llm = LLM(model_name="gemini-2.5-flash")
            res = llm.invoke(prompt)
            code = extract_code_block(res)
            output = run_generated_code(code)
            answer_prompt = ANSWER_PROMPT.format(
                question=state["query"],
                code=code,
                output=output
            )
            answer = llm.invoke(answer_prompt)
            state["result"] = answer

        except Exception as e:
            state["max_tries"] += 1
            state["critique"] = "loop"
            logger.info(f"Error occurred: {e}")
        
    return state


def ploting_code_agent(state: State):
    with st.status("📈 Running ploting_code_agent..."):
        date_time = datetime.datetime.now().isoformat()
        prompt = PLOT_PROMPT.format(
            question=state["query"],
            date_time=date_time
        )
        try:
            llm = LLM(model_name="gemini-2.5-flash")
            res = llm.invoke(prompt)
            code = extract_code_block(res)
            output = run_generated_code(code)
            state["result"] = output
            state["max_tries"] += 1

            chart_path = os.path.join(os.getcwd(), "data", "chart.png")
            if os.path.exists(chart_path):
                st.write("here is chart")
                st.image(chart_path, caption="📊 Generated Chart", use_column_width=True)
            else:
                st.warning("No chart found at ./data/chart.png")

        except Exception as e:
            state["max_tries"] += 1
            state["critique"] = "loop"
            logger.info(f"Error occurred: {e}")
    return state


def predict_using_ml(state: State):
    with st.status("🤖 Running predict agent..."):
        try:
            res = predict_model()
            state["result"] = res
        except Exception as e:
            state["max_tries"] += 1
            state["critique"] = "loop"
            logger.info(f"Error occurred: {e}")
    return state


def critique(state: State):
    with st.status("🔎 Running critique..."):
        prompt = CRITIQUE_PROMPT.replace("{question}", state["query"]) \
                        .replace("{answer}", state["result"])

        try:
            llm = LLM(model_name="gemini-2.5-flash")
            res = llm.invoke(prompt)
            logger.info("============================================", res)
            res_json = json.loads(repair_json(res))
            critique = res_json.get("critique")
            state["critique"] = critique
            state["max_tries"] += 1

        except Exception as e:
            state["max_tries"] += 1
            state["critique"] = "loop"
            logger.info(f"Error occurred: {e}")
    return state

def router(state: State):
    with st.status("🔀 Routing to the appropriate agent..."):
        return state
    


def analys_router(state: State):
    """
    Decide next node based on analysis_type predicted by LLM.
    """
    if state["analysis_type"] == "plot":
        return "plot"
    else:
        return "other"

def router_next(state: State):
    """
    Route depending on what was analyzed.
    """
    if state["analysis_type"] == "data":
        return "data_path"
    else:
        return "predict_path"

        
def semantic_cache_router(state: State):
    """
    Check cache before routing to router.
    """
    if state["from_cache"]:
        return "end"
    else:
        return "no"

def create_workflow_graph():
    workflow = StateGraph(State)

    workflow.add_node("semantic_cache", semantic_cache)
    workflow.add_node("analys", analys)
    workflow.add_node("data_scientist_agent", data_scientist_agent)
    workflow.add_node("ploting_code_agent", ploting_code_agent)
    workflow.add_node("predict", predict_using_ml)
    workflow.add_node("critique", critique)
    workflow.add_node("router", router)


        
    workflow.add_edge(START, "analys")


    workflow.add_conditional_edges(
        "analys",
        analys_router,
        {
            "plot": "ploting_code_agent",
            "other": "semantic_cache",
        },
    )


    workflow.add_conditional_edges(
        "semantic_cache",
        semantic_cache_router,
        {
            "end": END,
            "no": "router",
        },
    )



    workflow.add_conditional_edges(
        "router",
        router_next,
        {
            "data_path": "data_scientist_agent",
            "predict_path": "predict",
        },
    )

    workflow.add_edge("ploting_code_agent", END)
    workflow.add_edge("data_scientist_agent", "critique")
    workflow.add_conditional_edges(
        "critique",
        lambda state: state["critique"],
        {
            "stop": END,
            "loop": "data_scientist_agent",
        },
    )

    graph = workflow.compile()
    return graph