# """
# Main Application - Job Search and Application System
# """
# import os
# import json
# from dotenv import load_dotenv
# from tavily import TavilyClient
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Import modules
# from job_search import search_jobs
# from cover_letter_generator import generate_cover_letter
# from gmail_service import send_application_email
# from utils import save_application_materials

# # Load environment variables
# load_dotenv()

# def main():
#     """Main function to run the job search application."""
#     print("🚀 Job Search and Application System")
#     print("=" * 40)

#     # Initialize APIs
#     try:
#         # Initialize Tavily client
#         tavily_api_key = os.getenv('TAVILY_API_KEY')
#         if not tavily_api_key:
#             print("❌ TAVILY_API_KEY not found in environment variables")
#             return

#         tavily = TavilyClient(api_key=tavily_api_key)

#         # Initialize Gemini LLM
#         gemini_api_key = os.getenv('GEMINI_API_KEY')
#         if not gemini_api_key:
#             print("❌ GEMINI_API_KEY not found in environment variables")
#             return

#         llm = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             google_api_key=gemini_api_key,
#             temperature=0.7
#         )

#         print("✅ APIs initialized successfully")

#     except Exception as e:
#         print(f"❌ Error initializing APIs: {str(e)}")
#         return

#     # Get user input
#     print("\n📋 Please provide your information:")
#     skills = input("Enter your skills (comma-separated): ").split(',')
#     skills = [skill.strip() for skill in skills if skill.strip()]

#     resume_content = input("Paste your resume content (or press Enter to use sample): ")
#     if not resume_content.strip():
#         resume_content = "Experienced software developer with 5+ years in Python, JavaScript, and cloud technologies."

#     email = input("Enter your email for applications: ")

#     # Search for jobs
#     print(f"\n🔍 Searching for jobs matching: {', '.join(skills)}...")
#     jobs = search_jobs(tavily, skills)

#     if not jobs:
#         print("❌ No jobs found matching your skills")
#         return

#     # Display found jobs
#     print(f"\n📋 Found {len(jobs)} job opportunities:")
#     for i, job in enumerate(jobs, 1):
#         print(f"{i}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {job.get('location', 'N/A')}")

#     # Let user select a job
#     try:
#         choice = int(input(f"\nSelect a job to apply for (1-{len(jobs)}): "))
#         if choice < 1 or choice > len(jobs):
#             print("❌ Invalid selection")
#             return

#         selected_job = jobs[choice - 1]

#     except ValueError:
#         print("❌ Please enter a valid number")
#         return

#     # Generate cover letter
#     print(f"\n📝 Generating cover letter for {selected_job.get('title', 'position')}...")
#     cover_letter = generate_cover_letter(llm, selected_job, resume_content)

#     print("\n📄 Generated Cover Letter:")
#     print("=" * 50)
#     print(cover_letter)
#     print("=" * 50)

#     # Save application materials
#     filename = f"application_{selected_job.get('company', 'company').replace(' ', '_')}.txt"
#     save_result = save_application_materials(selected_job, cover_letter, filename)
#     print(f"\n{save_result}")

#     # Send application email
#     send_email = input("\n📧 Would you like to send this application via email? (y/n): ")
#     if send_email.lower() == 'y':
#         result = send_application_email(selected_job, cover_letter, email)
#         print(f"\n{result}")

#     print("\n🎉 Application process completed!")

# if __name__ == "__main__":
#     main()


"""
Job Search and Application System - Flask Blueprint
"""
import os
import json
import re
import base64
import uuid
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, send_file, current_app, render_template
from flask_cors import CORS
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

# Load environment variables
load_dotenv()

# Create blueprint
job_bp = Blueprint("job_bp", __name__, url_prefix="/api/jobs")

# Gmail Configuration
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


# ========== UTILITY FUNCTIONS ==========
def initialize_apis():
    """Initialize all required APIs"""
    try:
        # Initialize Tavily client
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found")

        tavily = TavilyClient(api_key=tavily_api_key)

        # Initialize Gemini LLM
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=gemini_api_key, temperature=0.7
        )

        return tavily, llm

    except Exception as e:
        current_app.logger.error(f"Error initializing APIs: {str(e)}")
        raise


def extract_company(content: str) -> str:
    """Extract company name from content."""
    patterns = [
        r"at\s+([A-Z][a-zA-Z\s&]+)(?:\s+Inc\.?|\s+LLC|\s+Corp\.?|\,|\s+in|$)",
        r"([A-Z][a-zA-Z\s&]+)\s+is hiring",
        r"([A-Z][a-zA-Z\s&]+)\s+job opportunity",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Unknown Company"


def extract_location(content: str) -> str:
    """Extract location from content."""
    patterns = [
        r"in\s+([A-Z][a-zA-Z\s\,]+)(?:\s+\.|\,|\s+\.)",
        r"Location:\s*([^\n]+)",
        r"based in\s+([^\n\.\,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Remote/Not specified"


def extract_salary(content: str) -> str:
    """Extract salary information from content."""
    patterns = [
        r"\$(\d{2,3}[,\d]{3,6})\s*[-–]\s*\$(\d{2,3}[,\d]{3,6})",
        r"salary[\s:]*\$?(\d{2,3}[,\d]{3,6})\s*[-–]\s*\$?(\d{2,3}[,\d]{3,6})",
        r"\$(\d{2,3}[,\d]{0,6})\s*(?:per year|annual|yr|hour|hr)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            if len(match.groups()) > 1:
                return f"${match.group(1)}-{match.group(2)}"
            else:
                return f"${match.group(1)}"
    return "Not specified"


def extract_requirements(content: str) -> List[str]:
    """Extract key requirements from content."""
    requirements = []
    requirement_keywords = ["requirements:", "qualifications:", "must have:", "skills:"]

    for keyword in requirement_keywords:
        if keyword.lower() in content.lower():
            start_idx = content.lower().find(keyword.lower()) + len(keyword)
            end_idx = content.find("\n\n", start_idx)
            if end_idx == -1:
                end_idx = min(start_idx + 500, len(content))

            requirement_text = content[start_idx:end_idx].strip()
            lines = requirement_text.split("\n")
            requirements.extend([line.strip("•- ") for line in lines if line.strip()])

    return requirements[:5] if requirements else ["See job description for details"]


def extract_job_description(content: str) -> str:
    """Extract job description text."""
    patterns = [
        r"Description[:\-]\s*(.*?)(?:\n\n|$)",
        r"Job Description[:\-]\s*(.*?)(?:\n\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return content[:400] + "..." if len(content) > 400 else content


def extract_email(content: str) -> str:
    """Extract recruiter/HR email from job content."""
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", content)
    return match.group(0) if match else "Not specified"


# def extract_job_info(search_result: Dict[str, Any]) -> Dict[str, Any]:
#     """Extract structured job information from search results."""
#     try:
#         content = search_result.get("content", "")
#         title = search_result.get("title", "")
#         url = search_result.get("url", "")

#         job_info = {
#             "id": str(uuid.uuid4()),
#             "title": title.strip() if title else "Not specified",
#             "company": extract_company(content),
#             "location": extract_location(content),
#             "description": extract_job_description(content),
#             "salary": extract_salary(content),
#             "requirements": extract_requirements(content),
#             "recruiter_email": extract_email(content),
#             "application_link": url,
#             "source_url": url,
#             "posted_date": datetime.now().isoformat(),
#         }

#         return job_info
#     except Exception as e:
#         current_app.logger.error(f"Error extracting job info: {e}")
#         return None


def extract_job_info(search_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured job information from search results."""
    try:
        content = search_result.get("content", "")
        title = search_result.get("title", "")
        url = search_result.get("url", "")

        # Improved company extraction
        company = extract_company(content)
        if company == "Unknown Company":
            # Try to extract from title
            title_parts = title.split(" - ")
            if len(title_parts) > 1 and " at " in title:
                company = title.split(" at ")[-1].split(" - ")[0].strip()
            elif " at " in title:
                company = title.split(" at ")[-1].strip()

        job_info = {
            "id": str(uuid.uuid4()),
            "title": title.strip() if title else "Not specified",
            "company": company,
            "location": extract_location(content),
            "description": extract_job_description(content),
            "salary": extract_salary(content),
            "requirements": extract_requirements(content),
            "recruiter_email": extract_email(content),
            "application_link": url,
            "source_url": url,
            "posted_date": datetime.now().isoformat(),
        }

        return job_info
    except Exception as e:
        current_app.logger.error(f"Error extracting job info: {e}")
        return None


# ========== JOB SEARCH FUNCTIONS ==========
def search_jobs(
    tavily: TavilyClient, skills: List[str], location: str = "", job_type: str = ""
) -> List[Dict[str, Any]]:
    """Search for jobs based on skills using Tavily search."""
    query = f"job openings for {', '.join(skills)}"
    if location:
        query += f" in {location}"
    if job_type:
        query += f" {job_type} positions"
    query += " latest job postings"

    try:
        response = tavily.search(query=query, max_results=10, include_raw_content=True)
        jobs = []

        for result in response.get("results", []):
            job_info = extract_job_info(result)
            if job_info:
                jobs.append(job_info)

        return jobs[:8]  # Return up to 8 jobs
    except Exception as e:
        current_app.logger.error(f"Error searching for jobs: {str(e)}")
        return []


# ========== COVER LETTER GENERATOR ==========
# def generate_cover_letter(
#     llm: ChatGoogleGenerativeAI,
#     job_info: Dict[str, Any],
#     resume_content: str,
#     tone: str = "professional",
# ) -> str:
#     """Generate a professional cover letter for the job application."""
#     tone_instruction = {
#         "professional": "Keep the tone professional and formal",
#         "enthusiastic": "Make the tone enthusiastic and energetic",
#         "concise": "Keep the tone concise and to the point",
#     }.get(tone, "Keep the tone professional")

#     prompt = f"""
#     Write a {tone} cover letter for the following job:

#     Job Title: {job_info.get('title', '')}
#     Company: {job_info.get('company', '')}
#     Job Description: {job_info.get('description', '')[:1000]}

#     Based on this resume information:
#     {resume_content}

#     {tone_instruction}. Highlight relevant skills and experience,
#     and keep it to about 250-300 words.
#     """

#     try:
#         response = llm.invoke(prompt)
#         return response.content
#     except Exception as e:
#         current_app.logger.error(f"Error generating cover letter: {str(e)}")
#         return f"Error generating cover letter: {str(e)}"


# ========== COVER LETTER GENERATOR ==========
def generate_cover_letter(
    llm: ChatGoogleGenerativeAI,
    job_info: Dict[str, Any],
    resume_content: str,
    tone: str = "professional",
) -> str:
    """Generate a professional cover letter for the job application."""
    tone_instruction = {
        "professional": "Write in a professional, formal business tone",
        "enthusiastic": "Write with enthusiasm and energy, showing excitement",
        "concise": "Write in a concise, direct manner without fluff",
    }.get(tone, "Write in a professional tone")

    # Improved prompt with clearer instructions
    prompt = f"""
    ACT AS AN EXPERIENCED CAREER COACH. Generate a compelling cover letter for a job application.
    
    JOB INFORMATION:
    - Position: {job_info.get('title', '')}
    - Company: {job_info.get('company', '')}
    - Description: {job_info.get('description', '')[:800]}
    
    CANDIDATE INFORMATION:
    {resume_content}
    
    REQUIREMENTS:
    1. {tone_instruction}
    2. Create a complete, ready-to-use cover letter (250-300 words)
    3. Address it to "Hiring Manager" if no specific name is available
    4. Highlight the most relevant skills from the resume that match the job
    5. Include a professional closing with placeholder for candidate name
    6. Do NOT include placeholders like [Your Name] - use "Candidate" temporarily
    7. Do NOT ask for more information - generate a complete letter
    
    IMPORTANT: Generate the entire cover letter without placeholders that need filling.
    """

    try:
        response = llm.invoke(prompt)
        content = response.content

        # Clean up the response if it contains any instructional text
        if (
            "here's a cover letter" in content.lower()
            or "dear hiring manager" in content.lower()
        ):
            # Extract just the cover letter part
            lines = content.split("\n")
            letter_lines = []
            in_letter = False

            for line in lines:
                if "dear" in line.lower() and (
                    "hiring" in line.lower() or "recruiter" in line.lower()
                ):
                    in_letter = True
                if in_letter:
                    letter_lines.append(line)

            if letter_lines:
                content = "\n".join(letter_lines)

        return content
    except Exception as e:
        current_app.logger.error(f"Error generating cover letter: {str(e)}")
        return f"Error generating cover letter: {str(e)}"


# ========== GMAIL SERVICE ==========
def get_gmail_service():
    """Get Gmail service using OAuth2 credentials"""
    creds = None
    token_path = "token.json"

    # Load existing token if available
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    client_secret = os.path.join(BASE_DIR, "client_secret.json")

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                current_app.logger.error(f"Token refresh failed: {e}")
                creds = None

        if not creds:
            # Prefer environment variable config
            client_config_str = os.getenv("GMAIL_CLIENT_CONFIG")
            if client_config_str:
                client_config = json.loads(client_config_str)
                temp_config_path = "temp_client_config.json"
                with open(temp_config_path, "w") as f:
                    json.dump(client_config, f)
                client_secret_file = temp_config_path
            else:
                client_secret_file = client_secret  # fallback

            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secret_file, SCOPES
                )

                # Run local server on fixed port (8080)
                creds = flow.run_local_server(
                    port=8080, redirect_uri_trailing_slash=True
                )

                # Save the credentials for the next run
                with open(token_path, "w") as token:
                    token.write(creds.to_json())

                # Clean up temp file
                if client_secret_file == temp_config_path:
                    os.remove(temp_config_path)

            except Exception as e:
                current_app.logger.error(f"Error initializing OAuth flow: {e}")
                return None

    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as e:
        current_app.logger.error(f"Error building Gmail service: {e}")
        return None


def create_message(sender, to, subject, body):
    """Create a message for an email."""
    message = f"From: {sender}\nTo: {to}\nSubject: {subject}\n\n{body}"
    return {"raw": base64.urlsafe_b64encode(message.encode()).decode()}


def send_message(service, user_id, message):
    """Send an email message."""
    try:
        message = (
            service.users().messages().send(userId=user_id, body=message).execute()
        )
        return message
    except Exception as e:
        current_app.logger.error(f"An error occurred: {e}")
        return None


# def send_application_email(
#     job_info: Dict[str, Any],
#     cover_letter: str,
#     recipient_email: str,
#     applicant_name: str,
# ) -> Dict[str, Any]:
#     """Send job application via email."""
#     try:
#         service = get_gmail_service()
#         if not service:
#             return {"success": False, "message": "Failed to initialize Gmail service"}

#         subject = f"Job Application: {job_info.get('title', 'Position')}"
#         body = f"""
#         Dear Hiring Manager,

#         Please find my application for the {job_info.get('title', '')} position at {job_info.get('company', '')}.

#         {cover_letter}

#         Thank you for considering my application. I look forward to the opportunity to discuss how my skills and experience can contribute to your team.

#         Best regards,
#         {applicant_name}

#         ---
#         Application for: {job_info.get('title', '')}
#         Company: {job_info.get('company', '')}
#         """

#         message = create_message("me", recipient_email, subject, body)
#         result = send_message(service, "me", message)

#         if result:
#             return {"success": True, "message": "Application email sent successfully!"}
#         else:
#             return {"success": False, "message": "Failed to send email"}

#     except Exception as e:
#         return {"success": False, "message": f"Error sending email: {str(e)}"}


def send_application_email(
    job_info: Dict[str, Any],
    cover_letter: str,
    recipient_email: str,
    applicant_name: str,
) -> Dict[str, Any]:
    """Send job application via email."""
    try:
        service = get_gmail_service()
        if not service:
            return {"success": False, "message": "Failed to initialize Gmail service"}

        # Clean the cover letter to remove any instructional text
        clean_cover_letter = cover_letter
        if "here's a cover letter" in clean_cover_letter.lower():
            # Extract just the letter part
            parts = clean_cover_letter.split("Dear Hiring Manager")
            if len(parts) > 1:
                clean_cover_letter = "Dear Hiring Manager" + parts[1]

        subject = f"Job Application: {job_info.get('title', 'Position')} at {job_info.get('company', 'Company')}"
        body = f"""
Dear Hiring Manager,

I am writing to express my interest in the {job_info.get('title', 'position')} position at {job_info.get('company', 'your company')} that I discovered through your recent posting.

{clean_cover_letter}

Thank you for considering my application. I have attached my resume for your review and would welcome the opportunity to discuss how my skills and experience can contribute to your team.

Best regards,
{applicant_name}

---
Application for: {job_info.get('title', '')}
Company: {job_info.get('company', '')}
        """

        message = create_message("me", recipient_email, subject, body)
        result = send_message(service, "me", message)

        if result:
            return {"success": True, "message": "Application email sent successfully!"}
        else:
            return {"success": False, "message": "Failed to send email"}

    except Exception as e:
        return {"success": False, "message": f"Error sending email: {str(e)}"}


# ========== API ROUTES ==========
@job_bp.route("/job_sk")
def jobs_index():
    """Render jobs UI page"""
    return render_template("feature4_ui/job_mailer.html")


@job_bp.route("/job_sk/search", methods=["POST"])
def api_search_jobs():
    """Search for jobs based on skills and filters"""
    try:
        data = request.get_json()
        skills = data.get("skills", [])
        location = data.get("location", "")
        job_type = data.get("job_type", "")

        if not skills:
            return jsonify({"error": "Skills are required"}), 400

        tavily, llm = initialize_apis()
        jobs = search_jobs(tavily, skills, location, job_type)

        return jsonify({"success": True, "count": len(jobs), "jobs": jobs})

    except Exception as e:
        current_app.logger.error(f"Error in job search: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@job_bp.route("/job_sk/generate-cover-letter", methods=["POST"])
def api_generate_cover_letter():
    """Generate a cover letter for a job application"""
    try:
        data = request.get_json()
        job_info = data.get("job", {})
        resume_content = data.get("resume", "")
        tone = data.get("tone", "professional")

        if not job_info or not resume_content:
            return jsonify({"error": "Job info and resume content are required"}), 400

        tavily, llm = initialize_apis()
        cover_letter = generate_cover_letter(llm, job_info, resume_content, tone)

        return jsonify({"success": True, "cover_letter": cover_letter})

    except Exception as e:
        current_app.logger.error(f"Error generating cover letter: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@job_bp.route("/job_sk/send-application", methods=["POST"])
def api_send_application():
    """Send job application via email"""
    try:
        data = request.get_json()
        job_info = data.get("job", {})
        cover_letter = data.get("cover_letter", "")
        email = data.get("email", "")
        applicant_name = data.get("name", "Job Applicant")

        if not job_info or not cover_letter or not email:
            return (
                jsonify({"error": "Job info, cover letter, and email are required"}),
                400,
            )

        result = send_application_email(job_info, cover_letter, email, applicant_name)

        return jsonify(result)

    except Exception as e:
        current_app.logger.error(f"Error sending application: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@job_bp.route("/job_sk/save-materials", methods=["POST"])
def api_save_materials():
    """Save application materials to a file"""
    try:
        data = request.get_json()
        job_info = data.get("job", {})
        cover_letter = data.get("cover_letter", "")

        if not job_info:
            return jsonify({"error": "Job info is required"}), 400

        # Create filename
        company = job_info.get("company", "company").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"application_{company}_{timestamp}.txt"
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

        # Save file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Job Application Materials\n")
            f.write(f"========================\n\n")
            f.write(f"Position: {job_info.get('title', '')}\n")
            f.write(f"Company: {job_info.get('company', '')}\n")
            f.write(f"Location: {job_info.get('location', '')}\n")
            f.write(
                f"Application Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(f"Cover Letter:\n")
            f.write(f"------------\n")
            f.write(cover_letter)
            f.write(f"\n\nJob Details:\n")
            f.write(f"-----------\n")
            f.write(job_info.get("description", ""))

        return jsonify(
            {
                "success": True,
                "message": f"Application materials saved to {filename}",
                "filename": filename,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error saving materials: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@job_bp.route("/job_sk/download/<filename>", methods=["GET"])
def download_file(filename):
    """Download application materials file"""
    try:
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        return send_file(filepath, as_attachment=True)

    except Exception as e:
        current_app.logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
