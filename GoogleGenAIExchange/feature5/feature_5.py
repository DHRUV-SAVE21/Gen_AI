# import os
# import re
# import json
# import uuid
# from typing import Dict, Any, List, Optional
# from flask import Blueprint, current_app, request, jsonify, render_template
# from dotenv import load_dotenv

# # LangGraph + LangChain Google Gemini wrapper
# from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Initialize LLM (LangChain wrapper)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.15
# )

# resume_generator_bp = Blueprint("resume_generator_bp", __name__)

# # ------------------------------
# # LangGraph pipeline nodes
# # ------------------------------


# # State shape: will be simple dict
# def parse_user_input(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Normalize input: split skills if provided as comma list, keep raw_text prompt.
#     Expected incoming state keys: 'raw_prompt' (str), 'target_skills' (str or list)
#     """
#     raw = state.get("raw_prompt", "")
#     # if user provided a comma separated skills string
#     ts = state.get("target_skills", "")
#     if isinstance(ts, str):
#         state["target_skills_list"] = [
#             s.strip().lower() for s in ts.split(",") if s.strip()
#         ]
#     else:
#         state["target_skills_list"] = [s.strip().lower() for s in (ts or [])]
#     state["raw_prompt"] = raw
#     return state


# # Template for the LLM: ask for structured JSON + resume text
# resume_prompt_template = PromptTemplate(
#     template="""
# You are ResumeForge, a professional resume builder assistant.
# Input: {input}

# Task:
# 1) Read the input and generate:
#    - A polished resume text (sections: Summary, Experience, Education, Skills, Projects).
#    - A structured JSON object with keys: name, title, contact, summary, experiences (list of {role, company, start, end, description}), education (list), skills (list), projects (list).
# 2) Output ONLY JSON with keys: "resume_text" (string), "resume_json" (object).
# Do NOT include any extra commentary.
# """,
#     input_variables=["input"],
# )


# def call_gemini_generate(state: Dict[str, Any]) -> Dict[str, Any]:
#     prompt_text = resume_prompt_template.format(input=state["raw_prompt"])
#     # Using ChatGoogleGenerativeAI.invoke - pass a chat-like payload (role+content)
#     # The wrapper returns an object with .content attribute (string).
#     response = llm.invoke([{"role": "user", "content": prompt_text}])
#     raw = response.content
#     state["raw_llm_output"] = raw
#     return state


# def parse_llm_output(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Extract the JSON blob from the model output and normalize."""
#     text = state.get("raw_llm_output", "")
#     # Extract the JSON object substring (first '{' to last '}')
#     start = text.find("{")
#     end = text.rfind("}")
#     if start == -1 or end == -1:
#         # fallback: try parse as JSON directly
#         try:
#             parsed = json.loads(text)
#         except Exception:
#             parsed = {"error": "Could not parse LLM output", "raw": text}
#         state["parsed"] = parsed
#         return state

#     json_text = text[start : end + 1]
#     try:
#         parsed = json.loads(json_text)
#     except Exception:
#         # sometimes the LLM nests JSON as strings; attempt to clean
#         try:
#             # replace single quotes, trailing commas (best-effort)
#             cleaned = json_text.replace("'", '"')
#             parsed = json.loads(cleaned)
#         except Exception:
#             parsed = {"error": "Could not parse LLM JSON", "raw": text}
#     state["parsed"] = parsed
#     # ensure resume_json and resume_text exist
#     state["resume_text"] = parsed.get("resume_text", "")
#     state["resume_json"] = parsed.get("resume_json", parsed)
#     return state


# def parse_resume_with_ai(resume_text: str) -> dict:
#     prompt = f"""
#     You are an ATS parser. Extract structured info from this resume:
#     {resume_text}

#     Return JSON in this schema:
#     {{
#       "name": "string",
#       "skills": ["skill1", "skill2"],
#       "experience": [{{"role": "string", "years": int}}],
#       "education": [{{"degree": "string", "institution": "string"}}]
#     }}
#     """

#     response = llm.invoke([{"role": "user", "content": prompt}])
#     text = response.content

#     # clean up JSON
#     text = re.sub(r"```(json)?", "", text).strip()
#     try:
#         return json.loads(text)
#     except Exception:
#         return {"error": "Failed to parse resume", "raw": text}


# def calculate_ats_score(parsed_resume: dict, target_role: str) -> int:
#     """Simple ATS score calculation based on skill match"""
#     target_skills = {
#         "ML Engineer": {"Python", "Machine Learning", "TensorFlow", "SQL"},
#         "Data Engineer": {"Python", "SQL", "ETL", "BigQuery"},
#     }

#     skills = set(parsed_resume.get("skills", []))
#     required = target_skills.get(target_role, set())

#     if not required:
#         return 50  # neutral if role unknown

#     matched = len(skills & required)
#     score = int((matched / len(required)) * 100)
#     return min(score, 100)


# def calculate_ats_and_visuals(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Calculate a simple ATS score:
#     - If user provided target_skills_list, % of those present in resume skills (case-insensitive).
#     Also build small visualization data:
#     - skill_match_count, missing_skills
#     - keyword frequency (top 10)
#     - experience timeline (for each experience count years)
#     """
#     resume_json = state.get("resume_json", {})
#     target_skills = state.get("target_skills_list", [])
#     # get skills from resume_json
#     skills = []
#     if isinstance(resume_json, dict):
#         skills = resume_json.get("skills", []) or []
#     # normalize
#     skills_lower = [s.lower() for s in skills]

#     matched = []
#     missing = []
#     for ts in target_skills:
#         if any(ts in sk for sk in skills_lower):
#             matched.append(ts)
#         else:
#             missing.append(ts)

#     ats_score = (
#         round((len(matched) / max(1, len(target_skills))) * 100, 2)
#         if target_skills
#         else None
#     )

#     # keyword frequency from resume_text
#     text = state.get("resume_text", "") or ""
#     words = [w.lower() for w in re.findall(r"\b[a-zA-Z\+\#0-9\-]+\b", text)]
#     freq = {}
#     for w in words:
#         if len(w) <= 2:
#             continue
#         freq[w] = freq.get(w, 0) + 1
#     # top words
#     top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:12]
#     keyword_freq = [{"word": k, "count": v} for k, v in top_words]

#     # experience timeline: sum durations if dates available (best-effort)
#     exp_list = (
#         resume_json.get("experiences", []) if isinstance(resume_json, dict) else []
#     )
#     timeline = []
#     for e in exp_list:
#         start = e.get("start")
#         end = e.get("end")
#         # best-effort year extraction
#         try:
#             sy = (
#                 int(start.split("-")[0])
#                 if start and "-" in start
#                 else (int(start) if start and start.isdigit() else None)
#             )
#         except Exception:
#             sy = None
#         try:
#             ey = (
#                 int(end.split("-")[0])
#                 if end and "-" in end
#                 else (int(end) if end and end.isdigit() else None)
#             )
#         except Exception:
#             ey = None
#         years = None
#         if sy and ey:
#             years = max(0, ey - sy)
#         timeline.append(
#             {
#                 "role": e.get("role", ""),
#                 "company": e.get("company", ""),
#                 "years": years or 0,
#             }
#         )

#     # small aggregate: projects count
#     projects = resume_json.get("projects", []) if isinstance(resume_json, dict) else []
#     projects_count = len(projects)

#     state["analysis"] = {
#         "ats_score": ats_score,
#         "matched_skills": matched,
#         "missing_skills": missing,
#         "keyword_freq": keyword_freq,
#         "timeline": timeline,
#         "projects_count": projects_count,
#         "resume_text": state.get("resume_text", ""),
#         "resume_json": resume_json,
#     }
#     return state


# # Build graph
# graph = StateGraph(dict)
# graph.add_node("parse_user_input", parse_user_input)
# graph.add_node("call_gemini_generate", call_gemini_generate)
# graph.add_node("parse_llm_output", parse_llm_output)
# graph.add_node("calculate_ats_and_visuals", calculate_ats_and_visuals)

# graph.set_entry_point("parse_user_input")
# graph.add_edge("parse_user_input", "call_gemini_generate")
# graph.add_edge("call_gemini_generate", "parse_llm_output")
# graph.add_edge("parse_llm_output", "calculate_ats_and_visuals")
# graph.add_edge("calculate_ats_and_visuals", END)

# agent = graph.compile()


# @resume_generator_bp.route("/builder")
# def index():
#     return render_template("feature5_ui/resume_generator.html")


# @resume_generator_bp.route("/builder/create", methods=["POST"])
# def create_resume():
#     try:
#         data = request.get_json(force=True)
#         # Assume frontend already generates `resume_text` via Gemini LangGraph pipeline
#         resume_text = data.get("resume_text")
#         target_role = data.get("target_role", "ML Engineer")

#         if not resume_text:
#             return jsonify({"error": "resume_text is required"}), 400

#         # Parse with AI
#         parsed = parse_resume_with_ai(resume_text)

#         # ATS score
#         ats_score = calculate_ats_score(parsed, target_role)

#         # Build charts dynamically
#         charts = {
#             "skills_distribution": {s: 1 for s in parsed.get("skills", [])},
#             "experience_vs_role": {
#                 exp["role"]: exp.get("years", 0) for exp in parsed.get("experience", [])
#             },
#             "ats_score_progress": [40, 60, ats_score],  # toy trend
#         }

#         return jsonify(
#             {
#                 "resume": resume_text,
#                 "ats_score": ats_score,
#                 "parsed": parsed,
#                 "charts": charts,
#             }
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


import os
import re
import json
import uuid
from typing import Dict, Any, List, Optional
from flask import Blueprint, current_app, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

# LangGraph + LangChain Google Gemini wrapper
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM (LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.15
)

resume_generator_bp = Blueprint("resume_generator_bp", __name__)

# ------------------------------
# LangGraph pipeline nodes
# ------------------------------

# State shape: will be simple dict
def parse_user_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize input: split skills if provided as comma list, keep raw_text prompt.
    Expected incoming state keys: 'raw_prompt' (str), 'target_skills' (str or list)
    """
    raw = state.get("raw_prompt", "")
    # if user provided a comma separated skills string
    ts = state.get("target_skills", "")
    if isinstance(ts, str):
        state["target_skills_list"] = [
            s.strip().lower() for s in ts.split(",") if s.strip()
        ]
    else:
        state["target_skills_list"] = [s.strip().lower() for s in (ts or [])]
    state["raw_prompt"] = raw
    return state


# Detailed prompt template for the LLM
resume_prompt_template = PromptTemplate(
    template="""
You are ResumeForge, a professional resume builder assistant. Your task is to create a comprehensive, professional resume based on the user's information.

USER INFORMATION:
{input}

INSTRUCTIONS:
1. Generate a polished, professional resume text with these sections:
   - Professional Summary
   - Work Experience (with detailed bullet points for each role)
   - Education
   - Skills (categorized if possible)
   - Projects (with descriptions and outcomes)
   - Certifications (if any)
   - Additional Sections (Languages, Awards, etc. if relevant)

2. Create a structured JSON object with these keys:
   - name: string
   - title: string
   - contact: object with email, phone, location, linkedin (if provided)
   - summary: string
   - experiences: list of objects with {role, company, location, start, end, description (bullet points)}
   - education: list of objects with {degree, institution, location, year, gpa (if provided)}
   - skills: list of objects with {category, items: []}
   - projects: list of objects with {name, description, technologies, outcomes}
   - certifications: list of objects with {name, issuer, date}
   - languages: list of objects with {language, proficiency}

3. For candidates with limited experience, focus on:
   - Education details and academic achievements
   - Relevant coursework
   - Personal projects and their impact
   - Transferable skills from other areas (volunteering, extracurriculars)
   - Strong summary emphasizing potential and eagerness to learn

4. Output ONLY JSON with keys: "resume_text" (string), "resume_json" (object).
Do NOT include any extra commentary outside the JSON.

IMPORTANT: Ensure the resume is tailored for the role and industry mentioned if provided.
""",
    input_variables=["input"],
)


def call_gemini_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = resume_prompt_template.format(input=state["raw_prompt"])
    # Using ChatGoogleGenerativeAI.invoke - pass a chat-like payload (role+content)
    # The wrapper returns an object with .content attribute (string).
    response = llm.invoke([{"role": "user", "content": prompt_text}])
    raw = response.content
    state["raw_llm_output"] = raw
    return state


def parse_llm_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the JSON blob from the model output and normalize."""
    text = state.get("raw_llm_output", "")
    # Extract the JSON object substring (first '{' to last '}')
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        # fallback: try parse as JSON directly
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"error": "Could not parse LLM output", "raw": text}
        state["parsed"] = parsed
        return state

    json_text = text[start : end + 1]
    try:
        parsed = json.loads(json_text)
    except Exception:
        # sometimes the LLM nests JSON as strings; attempt to clean
        try:
            # replace single quotes, trailing commas (best-effort)
            cleaned = json_text.replace("'", '"')
            # Remove trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            parsed = json.loads(cleaned)
        except Exception as e:
            parsed = {"error": f"Could not parse LLM JSON: {str(e)}", "raw": text}
    state["parsed"] = parsed
    # ensure resume_json and resume_text exist
    state["resume_text"] = parsed.get("resume_text", "")
    state["resume_json"] = parsed.get("resume_json", parsed)
    return state


def calculate_ats_and_visuals(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a simple ATS score:
    - If user provided target_skills_list, % of those present in resume skills (case-insensitive).
    Also build small visualization data:
    - skill_match_count, missing_skills
    - keyword frequency (top 10)
    - experience timeline (for each experience count years)
    """
    resume_json = state.get("resume_json", {})
    target_skills = state.get("target_skills_list", [])
    
    # Extract all skills from resume
    all_skills = []
    if isinstance(resume_json, dict):
        skills_data = resume_json.get("skills", [])
        if isinstance(skills_data, list):
            for item in skills_data:
                if isinstance(item, dict) and "items" in item:
                    all_skills.extend([s.lower() for s in item.get("items", [])])
                elif isinstance(item, str):
                    all_skills.append(item.lower())
                elif isinstance(item, dict) and "name" in item:
                    all_skills.append(item["name"].lower())
        elif isinstance(skills_data, str):
            all_skills = [s.strip().lower() for s in skills_data.split(",")]
    
    # Also check projects for technologies
    projects = resume_json.get("projects", []) if isinstance(resume_json, dict) else []
    for project in projects:
        if isinstance(project, dict) and "technologies" in project:
            techs = project["technologies"]
            if isinstance(techs, str):
                all_skills.extend([t.strip().lower() for t in techs.split(",")])
            elif isinstance(techs, list):
                all_skills.extend([t.lower() for t in techs if isinstance(t, str)])

    matched = []
    missing = []
    for ts in target_skills:
        if any(ts in sk for sk in all_skills):
            matched.append(ts)
        else:
            missing.append(ts)

    ats_score = (
        round((len(matched) / max(1, len(target_skills))) * 100, 2)
        if target_skills
        else None
    )

    # keyword frequency from resume_text
    text = state.get("resume_text", "") or ""
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z\+\#0-9\-]+\b", text)]
    freq = {}
    for w in words:
        if len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    # top words
    top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:12]
    keyword_freq = [{"word": k, "count": v} for k, v in top_words]

    # experience timeline: sum durations if dates available (best-effort)
    exp_list = (
        resume_json.get("experiences", []) if isinstance(resume_json, dict) else []
    )
    timeline = []
    for e in exp_list:
        if not isinstance(e, dict):
            continue
            
        start = e.get("start", "")
        end = e.get("end", "")
        # best-effort year extraction
        try:
            sy = (
                int(re.search(r'\d{4}', start).group())
                if start and re.search(r'\d{4}', start)
                else None
            )
        except Exception:
            sy = None
        try:
            ey = (
                int(re.search(r'\d{4}', end).group())
                if end and re.search(r'\d{4}', end)
                else (int(end) if end and end.isdigit() else None)
            )
        except Exception:
            ey = None
        years = None
        if sy and ey:
            years = max(0, ey - sy)
        elif sy and not end.lower().strip() in ["present", "current"]:
            years = 1  # Assume at least 1 year if only start date provided
        timeline.append(
            {
                "role": e.get("role", ""),
                "company": e.get("company", ""),
                "years": years or 0,
            }
        )

    # small aggregate: projects count
    projects_count = len(projects)

    state["analysis"] = {
        "ats_score": ats_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "keyword_freq": keyword_freq,
        "timeline": timeline,
        "projects_count": projects_count,
        "resume_text": state.get("resume_text", ""),
        "resume_json": resume_json,
    }
    return state


# Build graph
graph = StateGraph(dict)
graph.add_node("parse_user_input", parse_user_input)
graph.add_node("call_gemini_generate", call_gemini_generate)
graph.add_node("parse_llm_output", parse_llm_output)
graph.add_node("calculate_ats_and_visuals", calculate_ats_and_visuals)

graph.set_entry_point("parse_user_input")
graph.add_edge("parse_user_input", "call_gemini_generate")
graph.add_edge("call_gemini_generate", "parse_llm_output")
graph.add_edge("parse_llm_output", "calculate_ats_and_visuals")
graph.add_edge("calculate_ats_and_visuals", END)

agent = graph.compile()

# PDF Generation function
def generate_pdf(resume_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=9, spaceAfter=6))
    
    story = []
    
    # Title section
    if 'name' in resume_data:
        title_style = ParagraphStyle(
            name='Title',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            alignment=1
        )
        story.append(Paragraph(resume_data['name'], title_style))
    
    if 'title' in resume_data:
        story.append(Paragraph(resume_data['title'], styles['Heading2']))
        story.append(Spacer(1, 12))
    
    # Contact information
    if 'contact' in resume_data and resume_data['contact']:
        contact_info = []
        if isinstance(resume_data['contact'], dict):
            for key, value in resume_data['contact'].items():
                if value:
                    contact_info.append(f"{key}: {value}")
        elif isinstance(resume_data['contact'], str):
            contact_info.append(resume_data['contact'])
            
        if contact_info:
            story.append(Paragraph(" | ".join(contact_info), styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Summary
    if 'summary' in resume_data and resume_data['summary']:
        story.append(Paragraph("PROFESSIONAL SUMMARY", styles['Heading2']))
        story.append(Paragraph(resume_data['summary'], styles['BodyText']))
        story.append(Spacer(1, 12))
    
    # Experience
    if 'experiences' in resume_data and resume_data['experiences']:
        story.append(Paragraph("EXPERIENCE", styles['Heading2']))
        for exp in resume_data['experiences']:
            if isinstance(exp, dict):
                # Position and company
                position_text = f"<b>{exp.get('role', '')}</b>"
                if exp.get('company'):
                    position_text += f", {exp.get('company', '')}"
                if exp.get('location'):
                    position_text += f" - {exp.get('location', '')}"
                
                # Dates
                date_text = ""
                if exp.get('start') or exp.get('end'):
                    date_text = f"{exp.get('start', '')} - {exp.get('end', '')}"
                
                story.append(Paragraph(position_text, styles['Normal']))
                if date_text:
                    story.append(Paragraph(date_text, styles['Small']))
                
                # Description
                if exp.get('description'):
                    if isinstance(exp['description'], list):
                        for item in exp['description']:
                            story.append(Paragraph(f"• {item}", styles['BodyText']))
                    elif isinstance(exp['description'], str):
                        story.append(Paragraph(f"• {exp['description']}", styles['BodyText']))
                
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    
    # Education
    if 'education' in resume_data and resume_data['education']:
        story.append(Paragraph("EDUCATION", styles['Heading2']))
        for edu in resume_data['education']:
            if isinstance(edu, dict):
                edu_text = f"<b>{edu.get('degree', '')}</b>"
                if edu.get('institution'):
                    edu_text += f", {edu.get('institution', '')}"
                if edu.get('location'):
                    edu_text += f" - {edu.get('location', '')}"
                if edu.get('year'):
                    edu_text += f" ({edu.get('year', '')})"
                if edu.get('gpa'):
                    edu_text += f", GPA: {edu.get('gpa', '')}"
                
                story.append(Paragraph(edu_text, styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    
    # Skills
    if 'skills' in resume_data and resume_data['skills']:
        story.append(Paragraph("SKILLS", styles['Heading2']))
        
        if isinstance(resume_data['skills'], list):
            skills_text = ""
            for skill_item in resume_data['skills']:
                if isinstance(skill_item, dict) and 'category' in skill_item and 'items' in skill_item:
                    skills_text += f"<b>{skill_item['category']}:</b> {', '.join(skill_item['items'])}<br/>"
                elif isinstance(skill_item, str):
                    skills_text += f"{skill_item}, "
                elif isinstance(skill_item, dict) and 'name' in skill_item:
                    skills_text += f"{skill_item['name']}, "
            
            story.append(Paragraph(skills_text, styles['Normal']))
        elif isinstance(resume_data['skills'], str):
            story.append(Paragraph(resume_data['skills'], styles['Normal']))
        
        story.append(Spacer(1, 12))
    
    # Projects
    if 'projects' in resume_data and resume_data['projects']:
        story.append(Paragraph("PROJECTS", styles['Heading2']))
        for project in resume_data['projects']:
            if isinstance(project, dict):
                project_text = f"<b>{project.get('name', '')}</b>"
                if project.get('technologies'):
                    tech_text = ", ".join(project['technologies']) if isinstance(project['technologies'], list) else project['technologies']
                    project_text += f" | Technologies: {tech_text}"
                
                story.append(Paragraph(project_text, styles['Normal']))
                
                if project.get('description'):
                    desc = project['description']
                    if isinstance(desc, list):
                        for item in desc:
                            story.append(Paragraph(f"• {item}", styles['BodyText']))
                    else:
                        story.append(Paragraph(f"• {desc}", styles['BodyText']))
                
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    
    # Certifications
    if 'certifications' in resume_data and resume_data['certifications']:
        story.append(Paragraph("CERTIFICATIONS", styles['Heading2']))
        for cert in resume_data['certifications']:
            if isinstance(cert, dict):
                cert_text = f"<b>{cert.get('name', '')}</b>"
                if cert.get('issuer'):
                    cert_text += f", {cert.get('issuer', '')}"
                if cert.get('date'):
                    cert_text += f" ({cert.get('date', '')})"
                
                story.append(Paragraph(cert_text, styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    
    # Languages
    if 'languages' in resume_data and resume_data['languages']:
        story.append(Paragraph("LANGUAGES", styles['Heading2']))
        lang_text = ""
        for lang in resume_data['languages']:
            if isinstance(lang, dict):
                lang_text += f"{lang.get('language', '')} ({lang.get('proficiency', '')}), "
            elif isinstance(lang, str):
                lang_text += f"{lang}, "
        
        story.append(Paragraph(lang_text.rstrip(', '), styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


@resume_generator_bp.route("/builder")
def index():
    return render_template("feature5_ui/resume_generator.html")


@resume_generator_bp.route("/builder/create", methods=["POST"])
def create_resume():
    try:
        data = request.get_json(force=True)
        raw_prompt = data.get("raw_prompt", "")
        target_skills = data.get("target_skills", "")
        
        if not raw_prompt:
            return jsonify({"error": "Resume information is required"}), 400

        # Generate resume using LangGraph pipeline
        result = agent.invoke({
            "raw_prompt": raw_prompt,
            "target_skills": target_skills
        })
        
        # Generate a unique ID for this resume
        resume_id = str(uuid.uuid4())
        
        # Store the resume data (in a real app, you'd use a database)
        # For now, we'll just return it in the response
        
        return jsonify({
            "success": True,
            "id": resume_id,
            "analysis": result.get("analysis", {}),
            "resume_json": result.get("resume_json", {})
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@resume_generator_bp.route("/builder/download/<resume_id>", methods=["POST"])
def download_resume(resume_id):
    try:
        data = request.get_json(force=True)
        resume_data = data.get("resume_json", {})
        format_type = data.get("format", "pdf")
        
        if format_type == "pdf":
            pdf_buffer = generate_pdf(resume_data)
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=f"resume_{resume_id}.pdf",
                mimetype='application/pdf'
            )
        elif format_type == "json":
            json_str = json.dumps(resume_data, indent=2)
            return jsonify(resume_data)
        else:
            return jsonify({"error": "Unsupported format"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@resume_generator_bp.route("/builder/sample")
def get_sample_resume():
    """Return a sample resume structure for UI preview"""
    sample = {
        "name": "John Doe",
        "title": "Software Engineer",
        "contact": {
            "email": "john.doe@example.com",
            "phone": "(123) 456-7890",
            "location": "San Francisco, CA",
            "linkedin": "linkedin.com/in/johndoe"
        },
        "summary": "Experienced software engineer with 5+ years in full-stack development...",
        "experiences": [
            {
                "role": "Senior Developer",
                "company": "Tech Corp",
                "location": "San Francisco, CA",
                "start": "2020-01",
                "end": "Present",
                "description": [
                    "Led a team of 5 developers in building a scalable microservices architecture",
                    "Improved system performance by 40% through optimization techniques"
                ]
            }
        ],
        "education": [
            {
                "degree": "B.S. Computer Science",
                "institution": "University of California",
                "location": "Berkeley, CA",
                "year": "2018",
                "gpa": "3.8"
            }
        ],
        "skills": [
            {
                "category": "Programming Languages",
                "items": ["Python", "JavaScript", "Java", "C++"]
            },
            {
                "category": "Frameworks",
                "items": ["React", "Node.js", "Django", "Spring Boot"]
            }
        ],
        "projects": [
            {
                "name": "E-commerce Platform",
                "description": "Built a full-stack e-commerce solution serving 10,000+ users",
                "technologies": ["React", "Node.js", "MongoDB"],
                "outcomes": "Increased conversion rate by 25%"
            }
        ]
    }
    
    return jsonify(sample)