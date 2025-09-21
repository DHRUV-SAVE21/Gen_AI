import os
import json
import uuid
import fitz  # PyMuPDF
import re
import google.generativeai as genai
from flask import Blueprint, request, jsonify, render_template, current_app
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.0-flash"

resume_bp = Blueprint("resume_bp", __name__)


# -----------------------------
# Utilities
# -----------------------------
def save_uploaded_file(file_storage):
    UPLOAD_DIR = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file_storage.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(path)
    return path


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_parts = [page.get_text("text") for page in doc]
    return "\n".join(text_parts)


def extract_projects_from_text(text: str) -> list:
    projects = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if re.search(r"\bproject\b", line, re.I):
            block = []
            for j in range(i + 1, min(i + 8, len(lines))):
                block.append(lines[j])
            projects.append(" ".join(block))
    return list(dict.fromkeys(projects))[:10]


# -----------------------------
# Gemini Helpers
# -----------------------------
def extract_skills_dynamic(text: str) -> list:
    """Ask Gemini to extract skills from resume text."""
    prompt = f"""
Extract all technical and professional skills mentioned in the resume below.
Return them as a clean JSON array of strings. No extra text.

Resume:
{text[:4000]}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    raw = resp.text

    try:
        # Find JSON array in output
        if "[" in raw and "]" in raw:
            raw = raw[raw.index("[") : raw.rindex("]") + 1]
        return json.loads(raw)
    except Exception:
        return [s.strip("•- ") for s in raw.splitlines() if s.strip()]


def fetch_company_skills(company: str) -> list:
    """Ask Gemini what skills are needed at a given company."""
    prompt = f"""
List the 10 most important technical skills required for a Software Engineer role at {company}.
Return as bullet points only.
"""
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    return [line.strip("•- ") for line in resp.text.splitlines() if line.strip()]


def calculate_match(user_skills: list, company_skills: list) -> float:
    if not company_skills:
        return 0.0
    user_lower = [s.lower() for s in user_skills]
    matched = sum(1 for skill in company_skills if any(w in " ".join(user_lower) for w in skill.lower().split()))
    return round((matched / len(company_skills)) * 100, 2)


def compute_gap(user_skills: list, company_skills: list) -> list:
    user_lower = [s.lower() for s in user_skills]
    missing = [s for s in company_skills if not any(w in " ".join(user_lower) for w in s.lower().split())]
    return missing


# -----------------------------
# Routes
# -----------------------------
@resume_bp.route("/resume")
def index():
    """Dashboard page."""
    return render_template("feature3_ui/skill_gap.html")


@resume_bp.route("/resume/analyze", methods=["POST"])
def analyze_resume():
    """Upload resume → extract skills, projects."""
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["resume"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    path = save_uploaded_file(f)
    text = extract_text_from_pdf(path)
    skills = extract_skills_dynamic(text)
    projects = extract_projects_from_text(text)

    meta = {"resume_path": path, "skills": skills, "projects": projects}
    with open(path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return jsonify({"id": os.path.basename(path), "skills": skills, "projects": projects})


@resume_bp.route("/resume/company_analysis", methods=["POST"])
def company_analysis():
    """Compare resume skills vs target company requirements."""
    data = request.get_json() or {}
    rid = data.get("id")
    company = data.get("company", "").strip()

    if not rid or not company:
        return jsonify({"error": "Missing resume id or company"}), 400

    upload_dir = current_app.config.get("UPLOAD_FOLDER", "uploads")
    meta_path = os.path.join(upload_dir, rid + ".meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "Resume not found"}), 404

    with open(meta_path) as fh:
        meta = json.load(fh)

    user_skills = meta.get("skills", [])
    company_skills = fetch_company_skills(company)
    match_percent = calculate_match(user_skills, company_skills)
    gap = compute_gap(user_skills, company_skills)

    return jsonify(
        {
            "id": rid,
            "user_skills": user_skills,
            "company_skills": company_skills,
            "match_percent": match_percent,
            "skill_gap": gap,
            "projects": meta.get("projects", []),
        }
    )
