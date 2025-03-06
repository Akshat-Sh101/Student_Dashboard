from flask import Flask, render_template, request, redirect, session, jsonify
import mysql.connector
from hashlib import sha256
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "secret"

dataset = load_dataset("fazni/roles-based-on-skills")
skill_data = dataset['train']  

dataset_skills = [item['Role'] for item in skill_data]

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="1234", 
        database="StudentDashboard"
    )


def hash_password(password):
    return sha256(password.encode()).hexdigest()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        semester = request.form['semester']
        cgpa = request.form['cgpa']

        db = connect_db()
        cursor = db.cursor()

        hashed_pw = hash_password(password)  
        try:
            cursor.execute("INSERT INTO Students (name, email, password_hash, semester, cgpa) VALUES (%s, %s, %s, %s, %s)",
                           (name, email, hashed_pw, semester, cgpa))
            db.commit()
            return redirect('/login')
        except mysql.connector.Error as e:
            return f"Error: {e}"
        finally:
            cursor.close()
            db.close()
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        db = connect_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Students WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        db.close()
        
        if user and hash_password(password) == user['password_hash']:  
            session['user_id'] = user['id']  
            return redirect(f'/dashboard/{user["id"]}') 
        else:
            return "Invalid Email or Password"
    
    return render_template("login.html")

@app.route('/add_cgpa', methods=['POST'])
def add_cgpa():
    """Add a new CGPA entry for a semester"""
    data = request.json
    student_id = data.get("student_id")
    semester = data.get("semester")
    cgpa = data.get("cgpa")

    db = connect_db()
    cursor = db.cursor()

    try:
        cursor.execute("INSERT INTO CGPA (student_id, semester, cgpa) VALUES (%s, %s, %s)", 
                       (student_id, semester, cgpa))
        db.commit()
        return jsonify({"message": f"CGPA for Semester {semester} added successfully!"})
    except mysql.connector.Error as e:
        return jsonify({"message": f"Error: {e}"}), 500
    finally:
        cursor.close()
        db.close()

@app.route("/dashboard/<int:student_id>")
def dashboard(student_id):
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Fetch student info
    cursor.execute("SELECT * FROM Students WHERE id = %s", (student_id,))
    student = cursor.fetchone()

    if not student:
        return "Student not found!", 404

    # Fetch skills
    cursor.execute("SELECT skill FROM Skills WHERE student_id = %s", (student_id,))
    skills = [row["skill"] for row in cursor.fetchall()]

    # Fetch CGPA history
    cursor.execute("SELECT semester, cgpa FROM CGPA WHERE student_id = %s ORDER BY semester", (student_id,))
    cgpa_data = cursor.fetchall()

    cursor.close()
    db.close()

    # Extract semester and CGPA for the chart
    semesters = [row["semester"] for row in cgpa_data]
    cgpa_values = [row["cgpa"] for row in cgpa_data]
    curr_semester = max([row["semester"] for row in cgpa_data], default=1)
    # Calculate average CGPA
    avg_cgpa = round(sum(cgpa_values) / len(cgpa_values), 2) if cgpa_values else 0.0

    return render_template("dashboard.html", 
                           student=student, 
                           skills=skills, 
                           semesters=semesters, 
                           cgpa_values=cgpa_values, 
                           avg_cgpa=avg_cgpa,
                           curr_semester=curr_semester)

@app.route('/predict_cgpa', methods=['POST'])
def predict_cgpa():
    data = request.json
    student_id = data['student_id']
    
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT semester, cgpa FROM CGPA WHERE student_id = %s ORDER BY semester", (student_id,))
    cgpa_data = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) as skill_count FROM Skills WHERE student_id = %s", (student_id,))
    skill_data = cursor.fetchone()
    
    cursor.close()
    db.close()

    if not cgpa_data:
        return jsonify({"error": "Not enough data to predict CGPA"}), 400

    semesters = np.array([row["semester"] for row in cgpa_data]).reshape(-1, 1)
    cgpas = np.array([row["cgpa"] for row in cgpa_data])

    skill_count = skill_data['skill_count']
    X = np.hstack((semesters, np.full((len(semesters), 1), skill_count)))

    model = LinearRegression()
    model.fit(X, cgpas)

    next_semester = max(semesters)[0] + 1
    predicted_cgpa = model.predict([[next_semester, skill_count]])[0]

    return jsonify({"predicted_cgpa": round(predicted_cgpa, 2)})

@app.route('/add_skill', methods=['POST'])
def add_skill():
    data = request.json
    student_id = data.get('student_id')
    skill = data.get('skill')

    if not student_id or not skill:
        return jsonify({"error": "Missing student_id or skill"}), 400

    conn = connect_db()  
    cursor = conn.cursor()

    try:
        
        cursor.execute("SELECT 1 FROM Skills WHERE student_id = %s AND skill = %s", (student_id, skill))
        existing_skill = cursor.fetchone()

        if existing_skill:
            return jsonify({"message": "Skill already exists!"}), 200  # 🔥 FIXED: Changed status to 200

        
        cursor.execute("INSERT INTO Skills (student_id, skill) VALUES (%s, %s)", (student_id, skill))
        conn.commit()

        return jsonify({"message": "Skill added successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()


@app.route('/remove_skill', methods=['POST'])
def remove_skill():
    data = request.json
    student_id = data.get('student_id')
    skill = data.get('skill')

    if not student_id or not skill:
        return jsonify({"error": "Missing student_id or skill"}), 400

    conn = connect_db() 
    cursor = conn.cursor()

    try:
        
        cursor.execute("DELETE FROM Skills WHERE student_id = %s AND skill = %s", (student_id, skill))
        conn.commit()

        return jsonify({"message": "Skill removed successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/rate_skills', methods=['POST'])
def rate_skills():
    data = request.json
    student_id = data.get("student_id")

    if not student_id:
        return jsonify({"error": "Missing student_id"}), 400

    db = connect_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT skill FROM Skills WHERE student_id = %s", (student_id,))
    user_skills = [row["skill"] for row in cursor.fetchall()]

    cursor.close()
    db.close()

    if not user_skills:
        return jsonify({"message": "No skills found for the student"}), 404

    skill_vectors = {skill: np.zeros(len(dataset_skills)) for skill in dataset_skills}
    
    for i, skill in enumerate(dataset_skills):
        skill_vectors[skill][i] = 1

    user_vectors = np.mean([skill_vectors.get(skill, np.zeros(len(dataset_skills))) for skill in user_skills], axis=0)
    similarity_scores = {skill: cosine_similarity([user_vectors], [skill_vectors[skill]])[0][0] for skill in dataset_skills}

    suggested_skills = sorted(similarity_scores, key=similarity_scores.get, reverse=True)
    suggested_skills = [skill for skill in suggested_skills if skill not in user_skills][:5]  

    return jsonify({
        "current_skills": user_skills,
        "suggested_skills": suggested_skills
    })


HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

@app.route('/generate_roadmap', methods=['POST'])
def generate_roadmap():
    data = request.json
    skill = data.get("skill")

    if not skill:
        return jsonify({"error": "Missing skill"}), 400

    prompt = f"Create a detailed learning roadmap for mastering {skill}."

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500}, 
            "options": {"wait_for_model": True}  # Fix cold-start issues
        })

        
        if response.status_code != 200:
            return jsonify({"error": f"API Error: {response.status_code}, {response.text}"}), 500

        
        response_json = response.json()
        if not response_json or not isinstance(response_json, list):
            return jsonify({"error": "Invalid response format"}), 500

        roadmap = response_json[0].get('generated_text', "No roadmap generated")

        return jsonify({"roadmap": roadmap})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/logout')
def logout():
    session.pop('user_id', None)  
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
