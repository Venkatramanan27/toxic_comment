from pathlib import Path

import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, LSTM, TextVectorization
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "rnn_base" / "toxicity.h5"
TRAIN_DATA_CANDIDATES = [
    BASE_DIR / "data" / "raw" / "train.csv",
    BASE_DIR / "data_given" / "train.csv" / "train.csv",
]
MAX_FEATURES = 200000
SEQUENCE_LENGTH = 1800

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Toxic Comment Classifier</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 2rem; max-width: 800px; }
    textarea { width: 100%; min-height: 120px; padding: 0.75rem; }
    button { margin-top: 0.75rem; padding: 0.6rem 1rem; }
    table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
    th { background: #f5f5f5; }
  </style>
</head>
<body>
  <h1>Toxic Comment Classifier</h1>
  <p>Enter a comment and click Predict.</p>
  <form method="post">
    <textarea name="comment" placeholder="Type comment here...">{{ comment }}</textarea><br>
    <button type="submit">Predict</button>
  </form>
  {% if predictions %}
  <table>
    <tr><th>Label</th><th>Score</th><th>Flagged (&gt;= 0.5)</th></tr>
    {% for row in predictions %}
    <tr>
      <td>{{ row.label }}</td>
      <td>{{ "%.4f"|format(row.score) }}</td>
      <td>{{ "True" if row.flagged else "False" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}
</body>
</html>
"""


def _resolve_train_data() -> Path:
    for candidate in TRAIN_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find training data CSV.")


def _load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    train_path = _resolve_train_data()
    train_df = pd.read_csv(train_path)

    if "comment_text" not in train_df.columns:
        raise ValueError("Training data must include 'comment_text' column.")

    label_columns = [c for c in train_df.columns if c not in {"id", "comment_text"}]
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=SEQUENCE_LENGTH,
        output_mode="int",
    )
    vectorizer.adapt(train_df["comment_text"].astype(str).values)
    model = Sequential(
        [
            Embedding(MAX_FEATURES + 1, 32),
            Bidirectional(LSTM(32, activation="tanh")),
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(6, activation="sigmoid"),
        ]
    )
    model(tf.zeros((1, SEQUENCE_LENGTH), dtype=tf.int32))
    model.load_weights(str(MODEL_PATH))
    return model, vectorizer, label_columns


MODEL, VECTORIZER, LABELS = _load_assets()
app = Flask(__name__)


def predict_comment(comment: str):
    vectorized = VECTORIZER([comment])
    scores = MODEL.predict(vectorized, verbose=0)[0].tolist()
    return [
        {"label": label, "score": float(score), "flagged": bool(score >= 0.5)}
        for label, score in zip(LABELS, scores)
    ]


@app.route("/", methods=["GET", "POST"])
def home():
    comment = ""
    predictions = []
    if request.method == "POST":
        comment = request.form.get("comment", "").strip()
        if comment:
            predictions = predict_comment(comment)
    return render_template_string(HTML_TEMPLATE, comment=comment, predictions=predictions)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    comment = (payload.get("comment") or "").strip()
    if not comment:
        return jsonify({"error": "comment is required"}), 400
    return jsonify({"comment": comment, "predictions": predict_comment(comment)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
