import google.generativeai as genai
import os

# Configure Gemini
api_key=os.getenv("GEMINI_API_KEY")

def generate_mcqs(role, level, skills):
    """Generate MCQs dynamically using Gemini"""
    prompt = f"""
    Generate 5 multiple-choice questions for a mock interview.
    Role: {role}
    Level: {level}
    Candidate skills: {', '.join(skills)}
    
    Format strictly as JSON:
    [
      {{
        "question": "What is ...?",
        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
        "answer": "B"
      }}
    ]
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return eval(response.text)  # ⚠️ Safe only if you trust Gemini output

def run_mcq_session(role, level, skills):
    mcqs = generate_mcqs(role, level, skills)
    score = 0

    for i, q in enumerate(mcqs, start=1):
        print(f"\nQ{i}: {q['question']}")
        for opt in q["options"]:
            print(opt)
        ans = input("Your answer (A/B/C/D): ").strip().upper()
        if ans == q["answer"]:
            print("✅ Correct!")
            score += 1
        else:
            print(f"❌ Wrong. Correct answer: {q['answer']}")

    avg_score = (score / len(mcqs)) * 100
    return {"avg_score": avg_score}

def _cli():
    print("=== Mock Interview CLI (MCQ Mode) ===")
    role = input("Target role (e.g., Data Scientist): ").strip()
    level = input("Level (junior/mid/senior): ").strip() or "junior"
    skills = input("Your skills (comma separated): ").strip()
    user_skills = [s.strip().lower() for s in skills.split(",") if s.strip()]

    session = run_mcq_session(role, level, user_skills)

    print("\n\n=== SESSION SUMMARY ===")
    print(f"Your Score: {session['avg_score']:.1f}%")

if __name__ == "__main__":
    _cli()
