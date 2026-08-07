import os
import sqlite3
from pathlib import Path

from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "movie-review-secret"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "reviews.db"

try:
    import numpy as np
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    np = None
    imdb = None
    Dense = Embedding = SimpleRNN = Sequential = pad_sequences = None

VOCAB_SIZE = 10000
MAX_LENGTH = 200
MODEL_PATH = BASE_DIR / "imdb_sentiment_model.keras"

model = None


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(stored_password: str, password: str) -> bool:
    if not stored_password:
        return False
    if stored_password == password:
        return True
    try:
        return check_password_hash(stored_password, password)
    except ValueError:
        return False


def migrate_passwords() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users")
    rows = cur.fetchall()
    for user_id, stored_password in rows:
        if not stored_password or stored_password.startswith("pbkdf2:sha256:"):
            continue
        cur.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(stored_password), user_id))
    conn.commit()
    conn.close()


def build_imdb_model():
    global model
    if model is not None:
        return model

    if imdb is None or np is None:
        raise RuntimeError("TensorFlow is not available")

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_train = pad_sequences(x_train, maxlen=MAX_LENGTH)
    x_test = pad_sequences(x_test, maxlen=MAX_LENGTH)

    model = Sequential(
        [
            Embedding(VOCAB_SIZE, 32, input_length=MAX_LENGTH),
            SimpleRNN(32),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)
    model.save(MODEL_PATH)
    return model


def predict_sentiment(review: str) -> float:
    cleaned_review = review.lower().strip()
    if not cleaned_review:
        return 0.5

    try:
        model = build_imdb_model()
        word_index = imdb.get_word_index()
        words = cleaned_review.split()
        sequence = [word_index.get(word, 2) for word in words]
        padded = pad_sequences([sequence], maxlen=MAX_LENGTH)
        prob = model.predict(padded, verbose=0)[0][0]
        return float(prob)
    except Exception:
        # Lightweight fallback if TensorFlow or IMDb download is unavailable.
        positive_words = {"amazing", "excellent", "good", "great", "nice", "love", "beautiful", "best", "awesome", "fantastic", "joy", "happy", "wonderful", "super"}
        negative_words = {"bad", "boring", "terrible", "awful", "hate", "worst", "poor", "sad", "disappointing", "negative", "annoying", "weak", "junk"}
        tokens = [token.strip(".,!?:;()[]{}\"") for token in cleaned_review.split() if token.strip(".,!?:;()[]{}\"")]
        positive_hits = sum(1 for token in tokens if token in positive_words)
        negative_hits = sum(1 for token in tokens if token in negative_words)
        score = (positive_hits - negative_hits) / max(1, len(tokens))
        return float(max(0.0, min(1.0, score + 0.5)))


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT
        )
        """
    )
    cur.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in cur.fetchall()}
    if "email" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            movie TEXT NOT NULL,
            review TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            sentiment_label TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


init_db()
migrate_passwords()


@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    message = request.args.get("message", "")
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return render_template("login.html", error="Please enter both fields", message=message)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if row and verify_password(row[0], password):
            session["user"] = username
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password", message=message)
    return render_template("login.html", message=message)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not email or not password or not confirm_password:
            return render_template("signup.html", error="Please fill all fields")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hash_password(password)),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("signup.html", error="Username already exists")
        conn.close()
        return redirect(url_for("login", message="Account created successfully. Please login."))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/submit_reviews", methods=["POST"])
def submit_reviews():
    username = session.get("user")
    if not username:
        return redirect(url_for("login"))

    movies = ["Devara", "The Raja Saab", "Spider-Man: NWH"]
    reviews = []
    for movie in movies:
        review_text = request.form.get(f"review_{movie}", "").strip()
        if review_text:
            score = predict_sentiment(review_text)
            label = "Positive" if score >= 0.5 else "Negative"
            reviews.append((username, movie, review_text, score, label))

    if reviews:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO reviews (username, movie, review, sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            reviews,
        )
        conn.commit()
        conn.close()

    return redirect(url_for("results"))


@app.route("/results")
def results():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT movie, sentiment_label, COUNT(*) FROM reviews GROUP BY movie, sentiment_label ORDER BY movie"
    )
    rows = cur.fetchall()
    conn.close()

    summary = {}
    for movie, label, count in rows:
        summary.setdefault(movie, {"Positive": 0, "Negative": 0})
        summary[movie][label] = count

    return render_template("results.html", summary=summary)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)