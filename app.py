from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import sqlite3
import os
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model safely
model_path = os.path.join(os.path.dirname(__file__), "model.sav")
model=None
try:
    model = joblib.load(model_path)
    print("model loaded sucessfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
# Function to connect to SQLite database   
def connect_db():
    return sqlite3.connect("signup.db")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/logon")
def logon():
    return render_template("signup.html")

@app.route("/login")
def login():
    return render_template("signin.html")

@app.route("/result")
def result():
    return render_template("result.html")

'''@app.route("/predict")
def predict():
    return render_template("predict.html")'''


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("user")
        name = request.form.get("name")
        email = request.form.get("email")
        number = request.form.get("mobile")
        password = request.form.get("password")

        try:
            con = sqlite3.connect("signup.db")
            cur = con.cursor()
            cur.execute("SELECT * FROM info WHERE user = ?", (username,))
            existing_user = cur.fetchone()
            if existing_user:
                return render_template("signup.html", error="Username already exists. Please choose another one.")
            cur.execute("INSERT INTO info (user, email, password, mobile, name) VALUES (?, ?, ?, ?, ?)",  
                        (username, email, password, number, name))
            con.commit()
            con.close()
        except sqlite3.Error as e:
            print(f"Error inserting data into database: {e}")
            return render_template("signup.html", error="Database error. Please try again.")

        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        maill = request.form.get("user")
        password1 = request.form.get("password")

        try:
            con = sqlite3.connect("signup.db")
            cur = con.cursor()
            cur.execute("SELECT user, password FROM info WHERE user = ? AND password = ?", (maill, password1))
            data = cur.fetchone()
            con.close()
        except sqlite3.Error as e:
            print(f"Error querying database: {e}")
            return render_template("signin.html", error="Database error. Please try again.")

        if data:
            return redirect(url_for("index"))
        else:
            return render_template("signin.html", error="Invalid credentials")

    return render_template("signin.html")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        if model is None:
            return render_template("result.html", output="Model not loaded. Please check the model file.")
        try:
            # Extract only the 4 required features from the form
            int_features = [               
                float(request.form["time"]),
                float(request.form["instance_events_type"]),
                float(request.form["scheduling_class"]),
                float(request.form["priority"])
            ]

            final_features = np.array(int_features).reshape(1, -1)  # Ensure correct shape

            # Predict using the trained model
            prediction = model.predict(final_features)[0]

            # Define the output message based on prediction
            if prediction == 0:
                output = "The Task/Job is Not Failed and Data is Transferred Successfully!"
            else:
                output = "The Task/Job is Failed and Data Transfer was Unsuccessful!"
            graph_images = [
                "cnn_bilstm_metrics.png",
                "accuracy_comparison.png",
                "precision_comparison.png",
                "recall_comparison.png",
                "f1_score_comparison.png",
                
            ]  
            return render_template("result.html", output=output, graph_images=graph_images)

        except Exception as e:
            return render_template("predict.html", output=f"Error in prediction: {str(e)}")

    return render_template("predict.html")

@app.route("/notebook")
def notebook():
    return render_template("notebook.html")

@app.route("/about")
def about():
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True) 

