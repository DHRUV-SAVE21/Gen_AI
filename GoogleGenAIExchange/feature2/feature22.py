import os
import json
import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, render_template
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# -----------------------------
# Configuration
# -----------------------------
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAY_MAP = {day: i for i, day in enumerate(WEEKDAYS)}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# -----------------------------
# Flask Blueprint
# -----------------------------
scheduler_bp = Blueprint("scheduler_bp", __name__, template_folder="templates")


# ----------------------------- Gemini Functions -----------------------------
def clean_gemini_json(raw_text: str) -> str:
    if "```" in raw_text:
        parts = raw_text.split("```")
        for part in parts:
            if "{" in part and "}" in part:
                raw_text = part
                break
        if raw_text.strip().startswith("json"):
            raw_text = raw_text.strip()[4:]
    return raw_text.strip()


def call_gemini_for_plan(
    goal: str, availability: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    prompt = f"""
The user wants a personalized weekly roadmap.
Goal: {goal}
Weekly availability: {json.dumps(availability, indent=2)}

Task:
- Only suggest tasks for days with availability.
- Use start times from availability (first available slot).
- Give realistic tasks that progress step-by-step towards the goal.
- Output JSON strictly like this:

{{
  "mon": [{{"task": "string", "start": "HH:MM", "duration_min": int}}],
  "tue": [...],
  "wed": [...],
  "thu": [...],
  "fri": [...],
  "sat": [...],
  "sun": [...]
}}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    raw_text = response.text
    try:
        cleaned = clean_gemini_json(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        return {"error": f"Could not parse Gemini output: {e}", "raw": raw_text}


# ----------------------------- Google Calendar Functions -----------------------------
def get_calendar_service(client_secret_path: str = "feature2/client_secret.json"):
    creds = None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    client_secret_path = os.path.join(base_dir, client_secret_path)
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            os.path.join(base_dir, "token.json"), SCOPES
        )
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def next_weekday_date(start_date: datetime.date, target_weekday: int) -> datetime.date:
    days_ahead = (target_weekday - start_date.weekday() + 7) % 7
    return start_date + datetime.timedelta(days=days_ahead)


def schedule_plan_on_calendar(
    plan: Dict[str, List[Dict[str, Any]]],
    attendees: List[str] = [],
    timezone: str = "Asia/Kolkata",
    client_secret_path: str = "client_secret.json",
    week_start: Optional[str] = None,
):
    service = get_calendar_service(client_secret_path)
    results = []
    base_date = datetime.date.today()
    if week_start:
        try:
            base_date = datetime.datetime.strptime(week_start, "%Y-%m-%d").date()
        except:
            pass

    for day_code, tasks in plan.items():
        if not tasks or day_code not in WEEKDAY_MAP:
            continue
        target_date = next_weekday_date(base_date, WEEKDAY_MAP[day_code])

        for idx, task in enumerate(tasks, 1):
            task_name = task.get("task", f"Task {idx} ({day_code})")
            start_time = task.get("start", "09:00")
            duration_min = int(task.get("duration_min", 60))

            start_dt = datetime.datetime.combine(
                target_date, datetime.datetime.strptime(start_time, "%H:%M").time()
            )
            end_dt = start_dt + datetime.timedelta(minutes=duration_min)

            event = {
                "summary": task_name,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": timezone},
                "attendees": [
                    {"email": email.strip()} for email in attendees if email.strip()
                ],
                "reminders": {
                    "useDefault": False,
                    "overrides": [{"method": "popup", "minutes": 10}],
                },
            }

            created_event = (
                service.events()
                .insert(calendarId="primary", body=event, sendUpdates="all")
                .execute()
            )

            results.append(
                {
                    "task": task_name,
                    "date": target_date.isoformat(),
                    "start": start_time,
                    "duration_min": duration_min,
                    "gcal_id": created_event.get("id"),
                    "htmlLink": created_event.get("htmlLink"),
                }
            )

    return results


# ----------------------------- Routes -----------------------------
@scheduler_bp.route("/scheduler")
def index():
    return render_template("/feature2_ui/feature22.html")


@scheduler_bp.route("/scheduler/generate_plan", methods=["POST"])
def generate_plan():
    data = request.get_json()
    goal = data.get("goal", "")
    availability = data.get("availability", {})
    gemini_plan = call_gemini_for_plan(goal, availability)
    return jsonify({"plan": gemini_plan})


@scheduler_bp.route("/scheduler/schedule_plan", methods=["POST"])
def schedule_plan():
    data = request.get_json()
    plan = data.get("plan", {})
    week_start = data.get("week_start", None)
    scheduled_events = schedule_plan_on_calendar(plan, week_start=week_start)
    return jsonify(scheduled_events)
