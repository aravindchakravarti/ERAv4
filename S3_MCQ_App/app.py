import json
import re
from flask import Flask, request, jsonify, render_template
import os
import google.generativeai as genai

app = Flask(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_mcqs", methods=["POST"])
def generate_mcqs():
    data = request.json
    paragraph = data.get("text")

    if not paragraph:
        return jsonify({"error": "No text provided"}), 400

    # Stronger system prompt
    prompt = f"""
    Generate exactly 5 multiple-choice questions from the following text.
    Your response MUST be ONLY valid JSON (no explanations, no markdown).
    Use this format:
    {{
      "questions": [
        {{
          "question": "string",
          "options": ["string", "string", "string", "string"],
          "answer": "string"
        }}
      ]
    }}
    Text: {paragraph}
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    raw_text = response.text.strip()

    # Try parsing directly
    try:
        mcqs = json.loads(raw_text)
    except:
        # If extra text sneaks in, extract JSON with regex
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                mcqs = json.loads(match.group(0))
            except Exception as e:
                return jsonify({"error": f"JSON parse failed: {str(e)}", "raw": raw_text}), 500
        else:
            return jsonify({"error": "No JSON found in response", "raw": raw_text}), 500

    return jsonify(mcqs)

@app.route("/check_answers", methods=["POST"])
def check_answers():
    data = request.json
    user_answers = data.get("answers", [])
    correct_answers = data.get("correct_answers", [])

    score = sum(1 for ua, ca in zip(user_answers, correct_answers) if ua == ca)

    return jsonify({"score": score, "total": len(correct_answers)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# 