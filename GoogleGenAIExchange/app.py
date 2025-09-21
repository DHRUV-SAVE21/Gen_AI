from flask import Flask, render_template
from feature1.feature11 import career_bp1
from feature2.feature22 import scheduler_bp
from feature3.feature33 import resume_bp
from feature4.feature44 import job_bp
from feature5.feature_5 import resume_generator_bp
from chatbot.chatbot import aibot_bp

app = Flask(__name__)

app.register_blueprint(career_bp1, url_prefix="/feature1")
app.register_blueprint(scheduler_bp, url_prefix="/feature2")
app.register_blueprint(aibot_bp, url_prefix="/aibot")
app.register_blueprint(resume_bp, url_prefix="/resume_feature")
app.register_blueprint(job_bp, url_prefix="/job_mailer_feature")
app.register_blueprint(resume_generator_bp, url_prefix="/resume")


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
