# app.py
import streamlit as st
import joblib
import json
import pandas as pd
import re

# -------------------------
# Load model + metadata
# -------------------------
@st.cache_resource
def load_all():
    model = joblib.load("disease_prediction_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)

    # Clean feature names (strip leading/trailing whitespace)
    cleaned = [fn.strip() for fn in metadata["feature_names"]]
    metadata["feature_names"] = cleaned

    return model, label_encoder, metadata

model, le, meta = load_all()
SYMPTOM_LIST = meta["feature_names"]  # ordered list of features the model expects

# -------------------------
# Small utilities
# -------------------------
def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            delete = curr[j-1] + 1
            subst = prev[j-1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, subst))
        prev = curr
    return prev[-1]

def similarity(a: str, b: str) -> float:
    a, b = a or "", b or ""
    a = a.strip()
    b = b.strip()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))

# -------------------------
# Build variant map for symptom matching
# -------------------------
variant_map = {}
for s in SYMPTOM_LIST:
    base = normalize(s)
    variants = set([base, base.replace(" ", "")])
    parts = base.split()
    # add sub-phrases
    for i in range(len(parts)):
        for j in range(i+1, len(parts)+1):
            variants.add(" ".join(parts[i:j]))
    variant_map[s] = variants

# Manual important synonyms (add more here if you notice misses)
manual_synonyms = {
    "vomiting": ["throwing up", "puking", "vomit", "i vomited"],
    "nausea": ["nauseous", "feeling sick", "sick to my stomach"],
    "stomach_pain": ["stomach ache", "tummy ache", "belly ache", "abdominal pain"],
    "abdominal_pain": ["stomach ache", "belly pain", "tummy pain"],
    "yellowing_of_eyes": ["yellow eyes", "eyes are yellow", "jaundice"],
    "yellowish_skin": ["yellow skin", "skin is yellow"],
    "yellow_urine": ["dark urine", "yellow urine"],
    "fever": ["temperature", "feverish", "high temperature", "hot body"],
    "headache": ["head hurts", "migraine", "head pain"],
    "diarrhoea": ["loose motions", "watery stool", "diarrhea"],
    "cough": ["coughing", "persistent cough", "dry cough"],
    "dizziness": ["lightheaded", "feeling dizzy"],
    "itching": ["itchy", "skin itching", "i itch"],
    "skin_rash": ["rash", "rashes", "red spots on skin"],
    "breathlessness": ["shortness of breath", "difficulty breathing"],
    # add more if you find specific misses
}

# merge manual synonyms into variant_map's matching keys
for manual_k, syns in manual_synonyms.items():
    for s in SYMPTOM_LIST:
        if normalize(s) == normalize(manual_k):
            for syn in syns:
                variant_map[s].add(normalize(syn))

# -------------------------
# Symptom extraction
# -------------------------
def extract_symptoms_from_text(text: str) -> dict:
    """
    Returns dict: {symptom_name: 0/1}
    """
    text_n = normalize(text)
    words = text_n.split()
    detected = {s: 0 for s in SYMPTOM_LIST}

    # exact phrase matches for variants
    for s, variants in variant_map.items():
        for v in variants:
            pattern = r'\b' + re.escape(v) + r'\b'
            if re.search(pattern, text_n):
                detected[s] = 1
                break

    # fuzzy single-token matching and multi-token windows
    for s, variants in variant_map.items():
        if detected[s] == 1:
            continue
        for v in variants:
            v_words = v.split()
            if len(v_words) == 1:
                for w in words:
                    if similarity(w, v) >= 0.80:
                        detected[s] = 1
                        break
                if detected[s] == 1:
                    break
            else:
                L = len(v_words)
                for i in range(max(0, len(words) - L + 1)):
                    window = " ".join(words[i:i+L])
                    if similarity(window, v) >= 0.75:
                        detected[s] = 1
                        break
                if detected[s] == 1:
                    break

    # special heuristic: yellow + eye -> mark jaundice related fields
    if re.search(r'\byellow\b', text_n) and re.search(r'\beye|eyes|sclera|eyelid\b', text_n):
        for s in SYMPTOM_LIST:
            if "yellow" in normalize(s) or "jaundice" in normalize(s) or "yellowing" in normalize(s):
                detected[s] = 1

    # vomiting -> also mark nausea keys if present
    if re.search(r'\b(vomit|vomiting|throwing up|puking)\b', text_n):
        for s in SYMPTOM_LIST:
            if "nausea" in normalize(s) or "vomit" in normalize(s):
                detected[s] = 1

    return detected

# -------------------------
# Follow-up question system
# -------------------------
# Map follow-up questions to a list of symptom keys they should toggle to 1 on a 'yes'
FOLLOW_UP_QS = [
    {
        "id": "jaundice",
        "question": "Do your eyes or skin look yellow (yellowing of eyes/skin or dark urine)?",
        "symptoms_yes": ["yellowing_of_eyes", "yellowish_skin", "yellow_urine"]
    },
    {
        "id": "vomit_freq",
        "question": "Are you actively vomiting or have you vomited more than once today?",
        "symptoms_yes": ["vomiting", "nausea"]
    },
    {
        "id": "fever_high",
        "question": "Do you have a high fever (very hot, >38°C) or persistent fever?",
        "symptoms_yes": ["high_fever", "fever", "mild_fever"]
    },
    {
        "id": "cough_type",
        "question": "Do you have a persistent cough or are you coughing up phlegm?",
        "symptoms_yes": ["cough", "phlegm", "mucoid_sputum"]
    },
    {
        "id": "breathless_exertion",
        "question": "Do you feel breathless even while resting or with minimal exertion?",
        "symptoms_yes": ["breathlessness", "shortness_of_breath"]
    },
    {
        "id": "stool_water",
        "question": "Do you have watery stools or more than 3 loose stools a day?",
        "symptoms_yes": ["diarrhoea", "watery_stool"]  # watery_stool may not exist, but safe to include
    },
    {
        "id": "severe_headache",
        "question": "Is your headache very severe or accompanied by vision changes or vomiting?",
        "symptoms_yes": ["headache", "blurred_and_distorted_vision", "vomiting"]
    },
    # you can add more follow-ups mapping to symptoms from your metadata
]

# Helper: convert symptom keys to actual keys present in SYMPTOM_LIST (some names differ)
def resolve_symptom_keys(keys):
    resolved = []
    for k in keys:
        # try exact
        if k in SYMPTOM_LIST:
            resolved.append(k)
            continue
        # try normalized match
        matches = [s for s in SYMPTOM_LIST if normalize(s) == normalize(k)]
        if matches:
            resolved.extend(matches)
            continue
        # try substring
        matches = [s for s in SYMPTOM_LIST if normalize(k) in normalize(s) or normalize(s) in normalize(k)]
        resolved.extend(matches)
    return list(dict.fromkeys(resolved))  # unique preserving order

# Pre-resolve symptom lists for FOLLOW_UP_QS
for f in FOLLOW_UP_QS:
    f["symptoms_yes_resolved"] = resolve_symptom_keys(f["symptoms_yes"])

# -------------------------
# Prediction helpers
# -------------------------
def predict_and_probs(feature_dict):
    """
    feature_dict: {symptom_key: 0/1}
    returns: top3 list of tuples (disease_name, prob), all_probs dict mapping label->prob
    """
    # Build ordered dataframe
    row = [feature_dict.get(k, 0) for k in SYMPTOM_LIST]
    X = pd.DataFrame([row], columns=SYMPTOM_LIST)

    # Get probabilities (try predict_proba, else approximate)
    try:
        probs = model.predict_proba(X)[0]  # array over classes
        classes = model.classes_ if hasattr(model, "classes_") else range(len(probs))
    except Exception:
        # fallback: use decision_function if available, otherwise use predict and one-hot
        try:
            scores = model.decision_function(X)[0]
            # turn scores into softmax
            import math
            exps = [math.exp(s) for s in scores]
            ssum = sum(exps)
            probs = [e/ssum for e in exps]
        except Exception:
            preds = model.predict(X)[0]
            probs = [0.0]*len(le.classes_)
            probs[preds] = 1.0

    # map to label names using label encoder
    label_names = list(le.classes_)
    probs_by_label = {label_names[i]: float(probs[i]) for i in range(len(probs))}

    sorted_top = sorted(probs_by_label.items(), key=lambda x: x[1], reverse=True)
    return sorted_top[:3], probs_by_label

# -------------------------
# Streamlit UI / State
# -------------------------
st.set_page_config(page_title="Disease Prediction Chatbot (Top-3 + Follow-ups)", layout="wide")
st.title("💬 Disease Prediction Chatbot — Top-3 & Follow-ups")
st.write("Type your symptoms in plain English. The bot will extract symptoms, show top-3 disease probabilities, and ask follow-ups if uncertain.")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "current_features" not in st.session_state:
    st.session_state.current_features = {s: 0 for s in SYMPTOM_LIST}
if "followup_index" not in st.session_state:
    st.session_state.followup_index = 0
if "followup_asked" not in st.session_state:
    st.session_state.followup_asked = []  # ids asked
if "last_probs" not in st.session_state:
    st.session_state.last_probs = {}

# User input form
user_msg = st.text_input("You:", placeholder="E.g. I have yellow eyes and vomiting")

predict_clicked = st.button("Predict")

if predict_clicked and user_msg:
    # append to chat
    st.session_state.chat.append(("user", user_msg))

    # initial extraction
    extracted = extract_symptoms_from_text(user_msg)

    # merge into current_features (OR semantics: once a symptom detected it stays 1)
    for k, v in extracted.items():
        if v:
            st.session_state.current_features[k] = 1

    # run prediction
    top3, probs_by_label = predict_and_probs(st.session_state.current_features)
    st.session_state.last_probs = probs_by_label

    # display top3
    st.session_state.chat.append(("bot", f"Top-3 predictions: {', '.join([f'{t[0]} ({t[1]*100:.1f}%)' for t in top3])}"))

    # Decide if follow-up is needed:
    top_prob = top3[0][1] if top3 else 0.0
    second_prob = top3[1][1] if len(top3) > 1 else 0.0
    need_followup = False
    # Conditions for follow-up:
    if top_prob < 0.6 or (top_prob - second_prob) < 0.15:
        need_followup = True

    if need_followup:
        # select next follow-up question not asked yet and where symptoms mapped are currently 0
        chosen = None
        for f in FOLLOW_UP_QS:
            if f["id"] in st.session_state.followup_asked:
                continue
            # check if any resolved symptom for this followup is currently 0 (meaning missing)
            missing = [s for s in f["symptoms_yes_resolved"] if st.session_state.current_features.get(s, 0) == 0]
            if missing:
                chosen = f
                break

        if chosen:
            st.session_state.followup_index += 1
            st.session_state.followup_asked.append(chosen["id"])
            # ask question (yes/no)
            answer = st.radio(chosen["question"], options=["", "Yes", "No"], key=f"followup_{chosen['id']}")
            if answer == "Yes":
                # set resolved symptoms to 1
                for sym in chosen["symptoms_yes_resolved"]:
                    st.session_state.current_features[sym] = 1
                # re-run prediction immediately
                top3_new, probs_by_label_new = predict_and_probs(st.session_state.current_features)
                st.session_state.last_probs = probs_by_label_new
                st.session_state.chat.append(("bot", f"Thanks — updated Top-3: {', '.join([f'{t[0]} ({t[1]*100:.1f}%)' for t in top3_new])}"))
            elif answer == "No":
                st.session_state.chat.append(("bot", "Thanks — noted."))
            else:
                # If user hasn't answered yet, show the question area (we added radio)
                pass

        else:
            st.session_state.chat.append(("bot", "I asked all follow-ups I had — still uncertain. Showing top-3 results."))
    else:
        # confident enough; show final top3 and stop
        st.session_state.chat.append(("bot", f"Final Top-3: {', '.join([f'{t[0]} ({t[1]*100:.1f}%)' for t in top3])}"))

# Optionally, allow the user to press a 'Reset followups' button to start new session
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Start new session"):
        st.session_state.chat = []
        st.session_state.current_features = {s: 0 for s in SYMPTOM_LIST}
        st.session_state.followup_index = 0
        st.session_state.followup_asked = []
        st.session_state.last_probs = {}

with col2:
    if st.button("Show last probabilities"):
        if st.session_state.last_probs:
            sorted_probs = sorted(st.session_state.last_probs.items(), key=lambda x: x[1], reverse=True)[:10]
            st.write(pd.DataFrame(sorted_probs, columns=["Disease", "Probability"]))
        else:
            st.info("No prediction run yet.")

# Display chat history
st.markdown("---")
for sender, msg in st.session_state.chat:
    if sender == "user":
        st.markdown(f"🧍 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

# Debug / developer area
with st.expander("Debug: Symptom vector (detected)"):
    st.write(st.session_state.current_features)

with st.expander("Debug: Variant map sample (first 20 symptoms)"):
    sample = {k: list(v)[:6] for i,(k,v) in enumerate(variant_map.items()) if i < 20}
    st.write(sample)
