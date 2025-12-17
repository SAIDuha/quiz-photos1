from __future__ import annotations

from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import qrcode
import time
import socket
import os
import re
import math
import unicodedata

app = Flask(__name__)

# ------------------ Timing ------------------
QUESTION_DURATION = 20
REVEAL_DURATION   = 8
PODIUM_DURATION   = 8

# ------------------ Scoring (Kahoot-like, sans max) ------------------
SCORE_SCALE = 1200
SCORE_DECAY = 0.35
FAST_RATIO  = 0.20
SLOW_RATIO  = 0.85
SLOW_PENALTY = 120

STREAK_MUL_2 = 1.2
STREAK_MUL_3 = 1.5

ALLOW_LATE_JOIN = False

PHOTOS_DIR = os.path.join(app.root_path, "static", "photos")

# ------------------ Dropdown employés ------------------
# "Autre" doit exister (utilisé pour les blagues + choix joueur)
EMPLOYEES_ALL = [
  "Marc","Adélie","Annie","Maéva","Laure","Coline","Matthieu","Roberto","Romain","Serigne",
  "Manon","Mélanie","Said","Magid","Anais","Mathilde","Chang Mo","Nathalie","Camille","Fatima",
  "Nadège","Cathy","Julie","Guiseppe","Loric","Johanna","Férisa","Quentin","Edith","Thierry",
  "Corinne","Ophélie","Marie-Claude","Hanife","Alexia","Aline","Audrey","Cyndie","Aïda","Marie",
  "Elsa","constance","Valérie","sophie","Claire","Caroline","Eddie","Khalil",

  # Ajouts
  "Nicolas",
  "Véronique",
  "Lucie",
  "Mélanie Di martino",


  # Remplacement de "Aucun"
  "Autre"
]

# ------------------ Ordre du quiz (imposé) ------------------
# ✅ Eddie entre Laure et Lucie
# ✅ Blague 5 après Khalil
DESIRED_QUIZ_ORDER = [
    "Aline",
    "Mélanie Di martino",
    "Roberto",
    "Nicolas",
    "Marie Claude",
    "Blague 2",
    "Hanife",
    "Elsa",
    "Férisa",
    "Blague 1",
    "Romain",
    "Cathy",
    "Camille",
    "Matthieu",
    "Chang Mo",
    "Marie",
    "Véronique",
    "Fatima",
    "Blague 3",
    "Valérie",
    "Khalil",
    "Blague 5",
    "Adélie",
    "Nathalie",
    "Anaïs",
    "Thierry",
    "Annie",
    "Claire",
    "Coline",
    "Constance",
    "Said",
    "Corinne",
    "Magid",
    "Edith",
    "Blague 4",
    "Johanna",
    "Laure",
    "Eddie",
    "Lucie",
    "Serigne",
    "Maéva",
    "Marc",
    "Mélanie",
    "Nadège",
    "Sophie",
]

# ------------------ Proverbes ------------------
PROVERBS = [
  "Le temps dévoile les visages, pas les valeurs.",
  "Les souvenirs passent, la personnalité reste.",
  "On change de style, pas d’essence.",
  "Le temps n’efface pas, il transforme.",
  "Les années filent, les bons moments restent.",
  "La vie affine les contours.",
  "Grandir, c’est évoluer sans se trahir.",
  "Le présent éclaire le passé.",
  "Ce qu’on devient raconte ce qu’on a vécu.",
  "La meilleure version de soi se construit.",
  "Le temps révèle les détails qu’on oublie.",
  "Les années ajoutent du sens, pas seulement de l’âge.",
  "Rien ne se perd, tout se transforme.",
  "La constance vaut plus que l’apparence.",
  "Les étapes font la trajectoire.",
  "On ne revient pas en arrière, on comprend mieux.",
  "Le vrai changement, c’est la progression.",
  "Chaque année écrit une ligne de plus.",
  "L’expérience rend la suite plus claire.",
  "Le temps apprend ce que l’école n’enseigne pas.",
  "Les souvenirs sourient quand on les retrouve.",
  "Ce qui compte se reconnaît toujours.",
  "Le passé fait rire, le présent fait grandir.",
  "À force d’avancer, on devient soi.",
  "Le temps n’a pas d’ennemis, seulement des leçons.",
  "Les traits changent, l’énergie reste.",
  "Le chemin fait le caractère.",
  "On se découvre au fil du temps.",
  "Les années polissent, elles ne cassent pas.",
  "Ce qui est solide traverse les saisons.",
  "Le présent est un miroir du passé.",
  "Le temps révèle les plus belles surprises.",
  "Grandir, c’est gagner en impact.",
  "Les années passent, l’esprit demeure.",
  "Le temps donne du relief aux histoires.",
  "Le futur commence toujours maintenant.",
  "Chaque visage porte sa réussite."
]

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}

# ------------------ Clés détection (robustifiées) ------------------
CHILD_KEYS = [
    ("enfant", 0),
    ("bébé",  1),
    ("bebe",  1),
    ("petite",2),
    ("pre-ado",3),
    ("pré-ado",3),
    ("jeune", 4),
]

ADULT_KEYS = [
    ("adulte", 0),
    ("actuelle", 1),
    ("actuel", 2),
    ("maintenant", 3),
    ("aujourd", 4),
]

# ------------------ Normalisation noms ------------------
def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def employees_sorted() -> list[str]:
    other = "Autre"
    names = [x for x in EMPLOYEES_ALL if x != other]
    names_sorted = sorted(names, key=lambda s: _norm_name(s))
    return [other] + names_sorted

# ------------------ Utilitaires quiz auto ------------------
def _clean_base_name(stem: str) -> str:
    s = stem
    s = re.sub(r"\(\s*\d+\s*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\ben\s+couleur\b", "", s, flags=re.IGNORECASE)
    s = s.replace(" - ", " ")
    for k, _ in CHILD_KEYS + ADULT_KEYS:
        s = re.sub(rf"\b{k}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b20\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -_")
    return s.strip()

def _detect_kind(stem_lower: str) -> str | None:
    for k, _ in CHILD_KEYS:
        if k in stem_lower:
            return "child"
    for k, _ in ADULT_KEYS:
        if k in stem_lower:
            return "adult"
    return None

def _priority(stem_lower: str, kind: str) -> int:
    if kind == "child":
        for k, pr in CHILD_KEYS:
            if k in stem_lower:
                return pr
        return 999
    if kind == "adult":
        for k, pr in ADULT_KEYS:
            if k in stem_lower:
                return pr
        return 999
    return 999

def build_quiz_from_photos_folder() -> list[dict]:
    if not os.path.isdir(PHOTOS_DIR):
        return []

    buckets: dict[str, dict] = {}
    for fn in os.listdir(PHOTOS_DIR):
        path = os.path.join(PHOTOS_DIR, fn)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(fn)
        if ext not in VALID_EXT:
            continue

        stem = os.path.splitext(fn)[0]
        stem_lower = stem.lower()
        kind = _detect_kind(stem_lower)
        if kind is None:
            continue

        person = _clean_base_name(stem)
        if not person:
            continue

        pr = _priority(stem_lower, kind)
        if person not in buckets:
            buckets[person] = {"child": None, "adult": None}
        cur = buckets[person][kind]
        if cur is None or pr < cur[0]:
            buckets[person][kind] = (pr, fn)

    by_norm: dict[str, str] = {}
    for k in buckets.keys():
        by_norm[_norm_name(k)] = k

    quiz = []
    qid = 1
    for wanted in DESIRED_QUIZ_ORDER:
        wanted_norm = _norm_name(wanted)
        is_blague = wanted_norm.startswith("blague")

        bucket_key = by_norm.get(wanted_norm)
        if not bucket_key:
            continue

        child = buckets[bucket_key]["child"]
        adult = buckets[bucket_key]["adult"]
        if not child or not adult:
            continue

        quiz.append({
            "id": qid,
            "answer_label": "Autre" if is_blague else bucket_key,
            "image": child[1],
            "reveal_image": adult[1],
            "proverb": "" if is_blague else PROVERBS[(qid - 1) % len(PROVERBS)]
        })
        qid += 1

    return quiz

# ------------------ Scoring ------------------
def kahoot_like_score(answer_time_s: float) -> int:
    if answer_time_s < 0:
        answer_time_s = 0
    pts = SCORE_SCALE * math.exp(-SCORE_DECAY * answer_time_s)
    return int(round(pts))

def streak_multiplier(streak: int) -> float:
    if streak >= 3:
        return STREAK_MUL_3
    if streak == 2:
        return STREAK_MUL_2
    return 1.0

# ------------------ State ------------------
QUIZ = build_quiz_from_photos_folder()
current_question_index = 0

players: dict[str, dict] = {}
answers: dict[tuple, dict] = {}

PHASE_LOBBY    = "lobby"
PHASE_QUESTION = "question"
PHASE_REVEAL   = "reveal"
PHASE_PODIUM   = "podium"
PHASE_FINISHED = "finished"

phase = PHASE_LOBBY
phase_started_at = time.time()
game_started = False
game_paused = False
pause_started_at = None

# ------------------ Helpers ------------------
def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def get_join_url() -> str:
    return "https://quiz-photos1.onrender.com/join"


def current_question():
    return QUIZ[current_question_index]

def phase_duration(p: str) -> int:
    if p == PHASE_LOBBY:    return 0
    if p == PHASE_QUESTION: return QUESTION_DURATION
    if p == PHASE_REVEAL:   return REVEAL_DURATION
    if p == PHASE_PODIUM:   return PODIUM_DURATION
    return 0

def phase_remaining_seconds() -> int:
    d = phase_duration(phase)
    if d <= 0 or (not game_started) or game_paused:
        return 0
    elapsed = time.time() - phase_started_at
    return max(0, int(d - elapsed))

def answers_received_for_current_question() -> int:
    if not game_started or phase == PHASE_LOBBY or not QUIZ:
        return 0
    qid = current_question()["id"]
    return sum(1 for (_, qqid), _ in answers.items() if qqid == qid)

def compute_podium(top_n: int = 3):
    sorted_players = sorted(
        players.items(),
        key=lambda x: (
            -x[1]["score"],
            -x[1].get("fast_count", 0),
            x[1].get("total_time_correct_ms", 0)
        )
    )
    return [{
        "rank": i + 1,
        "pseudo": p,
        "score": info["score"],
        "fast": info.get("fast_count", 0),
        "streak": info.get("streak", 0),
    } for i, (p, info) in enumerate(sorted_players[:top_n])]

def build_heatmap_for_question(qid: int, bins: int = 10):
    if bins <= 0:
        bins = 10

    bucket_w = QUESTION_DURATION / bins
    buckets = [{"bin": i, "from_s": i*bucket_w, "to_s": (i+1)*bucket_w, "count": 0, "correct": 0} for i in range(bins)]

    for (_, qqid), a in answers.items():
        if qqid != qid:
            continue
        t_s = (a.get("t_ms", 0) / 1000.0)
        idx = int(min(bins - 1, max(0, t_s / bucket_w)))
        buckets[idx]["count"] += 1
        if a.get("correct"):
            buckets[idx]["correct"] += 1

    return buckets

def close_question_apply_no_answer_penalty():
    if not QUIZ:
        return
    qid = current_question()["id"]
    for pseudo in players.keys():
        if (pseudo, qid) not in answers:
            players[pseudo]["streak"] = 0

def maybe_advance():
    global phase, phase_started_at, current_question_index

    if not game_started or game_paused or not QUIZ:
        return
    if phase == PHASE_FINISHED:
        return
    if phase_remaining_seconds() > 0:
        return

    if phase == PHASE_QUESTION:
        close_question_apply_no_answer_penalty()
        phase = PHASE_REVEAL
        phase_started_at = time.time()
        return

    if phase == PHASE_REVEAL:
        phase = PHASE_PODIUM
        phase_started_at = time.time()
        return

    if phase == PHASE_PODIUM:
        if current_question_index < len(QUIZ) - 1:
            current_question_index += 1
            phase = PHASE_QUESTION
            phase_started_at = time.time()
        else:
            phase = PHASE_FINISHED
            phase_started_at = time.time()
        return

def force_next_phase():
    global phase, phase_started_at, current_question_index
    if not game_started or not QUIZ:
        return

    if phase == PHASE_QUESTION:
        close_question_apply_no_answer_penalty()
        phase = PHASE_REVEAL
        phase_started_at = time.time()
        return

    if phase == PHASE_REVEAL:
        phase = PHASE_PODIUM
        phase_started_at = time.time()
        return

    if phase == PHASE_PODIUM:
        if current_question_index < len(QUIZ) - 1:
            current_question_index += 1
            phase = PHASE_QUESTION
            phase_started_at = time.time()
        else:
            phase = PHASE_FINISHED
            phase_started_at = time.time()
        return

# ------------------ Pages ------------------
@app.route("/")
def index():
    return render_template("host.html", join_url=get_join_url(), allow_late_join=ALLOW_LATE_JOIN)

@app.route("/host")
def host():
    return render_template("host.html", join_url=get_join_url(), allow_late_join=ALLOW_LATE_JOIN)

@app.route("/join")
def join():
    return render_template("join.html")

@app.route("/qr")
def qr():
    join_url = get_join_url()
    img = qrcode.make(join_url)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# ------------------ API ------------------
@app.route("/api/game/start", methods=["POST"])
def api_game_start():
    global game_started, game_paused, phase, phase_started_at, current_question_index, answers, QUIZ

    QUIZ = build_quiz_from_photos_folder()

    if not QUIZ:
        return jsonify({"error": "Aucune paire photo détectée (enfant + adulte) dans static/photos."}), 400
    if len(players) == 0:
        return jsonify({"error": "Aucun joueur enregistré"}), 400

    for p in players:
        players[p]["score"] = 0
        players[p]["fast_count"] = 0
        players[p]["total_time_correct_ms"] = 0
        players[p]["streak"] = 0
        players[p]["last_award"] = 0

    answers = {}
    current_question_index = 0
    game_started = True
    game_paused = False
    phase = PHASE_QUESTION
    phase_started_at = time.time()
    return jsonify({"status": "ok", "questions": len(QUIZ)})

@app.route("/api/game/pause", methods=["POST"])
def api_game_pause():
    global game_paused, pause_started_at
    if not game_started:
        return jsonify({"error": "Le jeu n’est pas démarré"}), 400
    if not game_paused:
        game_paused = True
        pause_started_at = time.time()
    return jsonify({"status": "ok"})

@app.route("/api/game/resume", methods=["POST"])
def api_game_resume():
    global game_paused, phase_started_at, pause_started_at
    if not game_started:
        return jsonify({"error": "Le jeu n’est pas démarré"}), 400
    if game_paused and pause_started_at:
        paused_for = time.time() - pause_started_at
        phase_started_at += paused_for
    game_paused = False
    return jsonify({"status": "ok"})

@app.route("/api/game/next", methods=["POST"])
def api_game_next():
    if not game_started:
        return jsonify({"error": "Le jeu n’est pas démarré"}), 400
    force_next_phase()
    return jsonify({"status": "ok"})

@app.route("/api/game/reset", methods=["POST"])
def api_game_reset():
    global game_started, game_paused, phase, phase_started_at, current_question_index, players, answers
    game_started = False
    game_paused = False
    phase = PHASE_LOBBY
    phase_started_at = time.time()
    current_question_index = 0
    players = {}
    answers = {}
    return jsonify({"status": "ok"})

@app.route("/api/current_question")
def api_current_question():
    if game_started and not game_paused:
        maybe_advance()

    payload = {
        "game_started": game_started,
        "game_paused": game_paused,
        "phase": phase,
        "phase_remaining": phase_remaining_seconds(),
        "question_number": 0 if phase == PHASE_LOBBY else current_question_index + 1,
        "total_questions": len(QUIZ),
        "players_count": len(players),
        "players": sorted(players.keys()),
        "answers_received": answers_received_for_current_question(),
        "employees": employees_sorted(),
        "scoring": {
            "scale": SCORE_SCALE,
            "decay": SCORE_DECAY,
            "streak_mul_2": STREAK_MUL_2,
            "streak_mul_3": STREAK_MUL_3,
            "slow_penalty": SLOW_PENALTY
        }
    }

    if phase == PHASE_LOBBY or not QUIZ:
        return jsonify(payload)

    q = current_question()
    payload.update({
        "id": q["id"],
        "image_url": f"/static/photos/{q['image']}",
    })

    if phase in (PHASE_REVEAL, PHASE_PODIUM, PHASE_FINISHED):
        payload.update({
            "reveal_image_url": f"/static/photos/{q['reveal_image']}",
            "proverb": q.get("proverb", ""),
            "correct_label": q["answer_label"],
            "heatmap": build_heatmap_for_question(q["id"], bins=10)
        })

    if phase in (PHASE_PODIUM, PHASE_FINISHED):
        payload["podium"] = compute_podium(3)

    return jsonify(payload)

@app.route("/api/register_player", methods=["POST"])
def api_register_player():
    data = request.get_json(silent=True) or {}
    pseudo = (data.get("pseudo") or "").strip()

    if not pseudo:
        return jsonify({"error": "Pseudo requis"}), 400
    if game_started and not ALLOW_LATE_JOIN:
        return jsonify({"error": "Le jeu a déjà démarré (inscriptions fermées)."}), 400

    if pseudo not in players:
        players[pseudo] = {
            "score": 0,
            "joined_at": time.time(),
            "fast_count": 0,
            "total_time_correct_ms": 0,
            "streak": 0,
            "last_award": 0
        }

    return jsonify({"status": "ok"})

@app.route("/api/submit_answer", methods=["POST"])
def api_submit_answer():
    if game_started and not game_paused:
        maybe_advance()

    if not game_started or phase != PHASE_QUESTION or game_paused or not QUIZ:
        return jsonify({"error": "Réponses fermées."}), 400

    data = request.get_json(silent=True) or {}
    pseudo = (data.get("pseudo") or "").strip()
    chosen_label = (data.get("choice_label") or "").strip()

    if not pseudo or pseudo not in players:
        return jsonify({"error": "Joueur non enregistré"}), 400
    if not chosen_label:
        return jsonify({"error": "Choix requis"}), 400
    if chosen_label not in EMPLOYEES_ALL:
        return jsonify({"error": "Choix invalide"}), 400

    q = current_question()
    key = (pseudo, q["id"])
    if key in answers:
        return jsonify({"status": "already_answered"}), 200

    now = time.time()
    answer_time_s = now - phase_started_at
    t_ms = int(answer_time_s * 1000)

    correct = (chosen_label == q["answer_label"])
    points = 0
    new_streak = players[pseudo]["streak"]

    if correct:
        new_streak += 1
        mul = streak_multiplier(new_streak)

        base_pts = kahoot_like_score(answer_time_s)

        if answer_time_s >= (QUESTION_DURATION * SLOW_RATIO):
            base_pts = max(0, base_pts - SLOW_PENALTY)

        points = int(round(base_pts * mul))

        players[pseudo]["score"] += points
        players[pseudo]["streak"] = new_streak
        players[pseudo]["last_award"] = points

        if answer_time_s <= (QUESTION_DURATION * FAST_RATIO):
            players[pseudo]["fast_count"] += 1

        players[pseudo]["total_time_correct_ms"] += t_ms

    else:
        players[pseudo]["streak"] = 0
        players[pseudo]["last_award"] = 0

    answers[key] = {
        "choice_label": chosen_label,
        "ts": now,
        "t_ms": t_ms,
        "correct": correct,
        "points": points
    }

    return jsonify({
        "status": "ok",
        "correct": correct,
        "points_awarded": points,
        "total_score": players[pseudo]["score"],
        "streak": players[pseudo]["streak"],
        "answer_time_ms": t_ms
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
