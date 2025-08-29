# app.py
from flask import Flask, jsonify, render_template
import random, time

app = Flask(__name__)

# ---- ROUTES ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics")
def metrics():
    # simulate metrics (you will replace this with real training logs later)
    data = {
        "epoch": int(time.time()) % 100,              # fake epoch
        "train_loss": random.uniform(0.1, 1.0),       # fake loss
        "val_loss": random.uniform(0.1, 1.0),
        "accuracy": random.uniform(0.5, 1.0),
        "lr": random.choice([0.001, 0.0005, 0.0001]),
        "time_per_epoch": random.uniform(20, 60),
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
