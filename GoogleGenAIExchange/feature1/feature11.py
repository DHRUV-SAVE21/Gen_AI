# from typing import List, Dict, Any
# from typing_extensions import TypedDict
# import os, json
# from dotenv import load_dotenv
# from flask import Blueprint, request, jsonify, render_template
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from tavily import TavilyClient

# load_dotenv()

# gemini_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2
# )
# tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# def tavily_search(query: str, max_results: int = 5) -> list:
#     """Search Tavily for resources (articles, tutorials, docs)."""
#     results = tavily.search(query=query, max_results=max_results)
#     return [
#         {"title": r["title"], "url": r["url"]} for r in results["results"][:max_results]
#     ]

# class State(TypedDict):
#     user_input: str
#     user_skills: List[str]
#     raw_llm_output: str
#     jobs_data: Dict[str, Any]


# prompt_template = PromptTemplate(
#     template="""
# You are CareerSage, an AI career advisor.

# Input: {input}.

# Task:
# 1) Suggest 3–5 suitable career paths.
# 2) For each, provide:
#    - Job title
#    - Short description
#    - Missing skills
#    - Free online learning resources (at least one per skill)
# 3) Return ONLY JSON with key "jobs" (list of job objects).
# """,
#     input_variables=["input"],
# )


# def generate_careers(user_input: str) -> Dict[str, Any]:
#     """Generate careers, enrich with Tavily resources, and return structured data."""

#     prompt = prompt_template.format(input=user_input)
#     response = gemini_llm.invoke([{"role": "user", "content": prompt}])
#     raw = response.content
#     start, end = raw.find("{"), raw.rfind("}") + 1
#     jobs_json = raw[start:end]
#     jobs_data = json.loads(jobs_json)

#     for job in jobs_data.get("jobs", []):

#         skills = []
#         for s in job.get("missing_skills", []):
#             if isinstance(s, str):
#                 skills.append(s.lower())
#             elif isinstance(s, dict) and "skill" in s:
#                 skills.append(s["skill"].lower())
#         job["missing_skills"] = skills

#         job["resources"] = {
#             s: tavily_search(f"free learning resources for {s}") for s in skills
#         }

#         roadmap_text = []
#         day = 1
#         for skill in skills:
#             roadmap_text.append(
#                 {
#                     "day_range": f"Day {day}-{day+3}",
#                     "focus": skill,
#                     "resources": job["resources"].get(skill, []),
#                 }
#             )
#             day += 4
#         job["roadmap"] = roadmap_text

#     return jobs_data

# career_bp1 = Blueprint("career_bp", __name__)


# @career_bp1.route("/feat1")
# def index():
#     return render_template("feature1_ui/feature11.html")


# @career_bp1.route("/feat1/get_careers", methods=["POST"])
# def get_careers():
#     data = request.get_json()
#     skills = data.get("skills", "")
#     jobs_data = generate_careers(skills)
#     return jsonify(jobs_data)


# def run_career_pipeline(user_input: str) -> Dict[str, Any]:
#     """Main pipeline for Flask. Returns jobs list with details."""
#     return generate_careers(user_input)


from typing import List, Dict, Any
from typing_extensions import TypedDict
import os, json
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2
)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def tavily_search(query: str, max_results: int = 5) -> list:
    """Search Tavily for resources (articles, tutorials, docs)."""
    results = tavily.search(query=query, max_results=max_results)
    return [
        {"title": r["title"], "url": r["url"]}
        for r in results["results"][:max_results]
    ]

class State(TypedDict):
    user_input: str
    user_skills: List[str]
    raw_llm_output: str
    jobs_data: Dict[str, Any]


prompt_template = PromptTemplate(
    template="""
You are CareerSage, an AI career advisor.

Input: {input}.

Task:
1) Suggest 3–5 suitable career paths.
2) For each, provide:
   - Job title
   - Short description
   - Missing skills
3) Return ONLY JSON with key "jobs" (list of job objects).
""",
    input_variables=["input"],
)


def parse_input(state: State) -> State:
    state["user_skills"] = [s.strip().lower() for s in state["user_input"].split(",")]
    return state


def generate_llm_output(state: State) -> State:
    prompt = prompt_template.format(input=state["user_input"])
    response = gemini_llm.invoke([{"role": "user", "content": prompt}])
    state["raw_llm_output"] = response.content
    return state


def enrich_jobs(state: State) -> State:
    raw = state["raw_llm_output"]
    start, end = raw.find("{"), raw.rfind("}") + 1
    jobs_json = raw[start:end]
    jobs_data = json.loads(jobs_json)

    user_skills = state["user_skills"]

    for job in jobs_data.get("jobs", []):
        skills = []
        for s in job.get("missing_skills", []):
            if isinstance(s, str):
                skills.append(s.lower())
            elif isinstance(s, dict) and "skill" in s:
                skills.append(s["skill"].lower())
        job["missing_skills"] = skills

        matched = len([s for s in user_skills if s not in skills])
        total = matched + len(skills)
        job["match_percent"] = round((matched / total) * 100, 2) if total > 0 else 0

        job["resources"] = {
            s: tavily_search(f"free learning resources for {s}") for s in skills
        }

        roadmap_text = []
        day = 1
        for skill in skills:
            roadmap_text.append(
                {
                    "day_range": f"Day {day}-{day+3}",
                    "focus": skill,
                    "resources": job["resources"].get(skill, []),
                }
            )
            day += 4
        job["roadmap"] = roadmap_text

    state["jobs_data"] = jobs_data
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("parse_input", parse_input)
graph_builder.add_node("generate_llm_output", generate_llm_output)
graph_builder.add_node("enrich_jobs", enrich_jobs)

graph_builder.set_entry_point("parse_input")
graph_builder.add_edge("parse_input", "generate_llm_output")
graph_builder.add_edge("generate_llm_output", "enrich_jobs")
graph_builder.add_edge("enrich_jobs", END)

career_agent = graph_builder.compile()

career_bp1 = Blueprint("career_bp1", __name__)


@career_bp1.route("/feat1")
def index():
    return render_template("feature1_ui/feature11.html")


@career_bp1.route("/feat1/get_careers", methods=["POST"])
def get_careers():
    data = request.get_json()
    skills = data.get("skills", "")
    final_state = career_agent.invoke({"user_input": skills})
    return jsonify(final_state["jobs_data"])
