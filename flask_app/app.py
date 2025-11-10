import matplotlib
matplotlib.use("Agg") 
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Setup
load_dotenv()
app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": ["chrome-extension://*", "http://127.0.0.1:5000"]}})

@app.route("/get_youtube_key", methods=["GET"])
def get_youtube_key():
    return jsonify({"api_key": os.getenv("YOUTUBE_DATA_API_V3")})

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
MODEL_NAME = "FinalModel"
MODEL_VERSION = 3     

# Load model from MLflow registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model from MLflow registry: {model_uri}")

# Text preprocessing (same as training)
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
        stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        comment = " ".join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return comment

# Routes
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Sentiment Analysis API!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    comments = data.get("comments", [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        preds = model.predict(pd.Series(preprocessed))
        response = [{"comment": c, "sentiment": int(p)} for c, p in zip(comments, preds)]
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts", {})
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments", [])
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data", [])
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        for s in [-1, 0, 1]:
            if s not in monthly_percentages.columns:
                monthly_percentages[s] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        for s in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[s],
                marker="o",
                label=labels[s],
                color=colors[s],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
