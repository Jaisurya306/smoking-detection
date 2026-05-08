import streamlit as st
import cv2
import numpy as np
import base64
import smtplib
from email.message import EmailMessage
from datetime import datetime
from inference_sdk import InferenceHTTPClient

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Smoking Detection",
    page_icon="🚭",
    layout="wide"
)

# ===============================
# LOGIN CONFIG
# ===============================
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ===============================
# SESSION STATE
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "run" not in st.session_state:
    st.session_state.run = False
if "last_person" not in st.session_state:
    st.session_state.last_person = None
if "last_fine_time" not in st.session_state:
    st.session_state.last_fine_time = None

# ===============================
# GLOBAL CSS
# ===============================
st.markdown("""
<style>
.stApp { background:#fffef0; color:#1f2937; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffff7f, #f1fff9);
}
.card {
    background:white;
    border-radius:18px;
    padding:22px;
    box-shadow:0 15px 35px rgba(0,0,0,0.12);
}
.success { border-left:6px solid #22c55e; }
.danger { border-left:6px solid #ef4444; }

button {
    background: linear-gradient(135deg,#3b82f6,#2563eb) !important;
    color:white !important;
    border-radius:10px !important;
    font-weight:600 !important;
    
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOGIN PAGE
# ===============================
def login_page():

    st.markdown("""
    <style>
    .login-title {
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        color: #b89a20;
        margin-bottom: 25px;
    }

    div[data-baseweb="input"] input {
        background: #fff8b0 !important;
        border-radius: 10px !important;
        padding: 12px !important;
        border: none !important;
        font-size: 16px;
    }

    button {
        background: #d6c15f !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        height: 45px;
    }
    </style>
    """, unsafe_allow_html=True)

    # CENTERED SMALL WIDTH FORM
    left, center, right = st.columns([2, 1.2, 2])

    with center:
     
        st.markdown('<div class="login-title">ADMIN LOGIN</div>', unsafe_allow_html=True)

        username = st.text_input("user")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("✅ Login Successful")
                st.rerun()
            else:
                st.error("❌ Invalid Credentials")

        st.markdown('</div>', unsafe_allow_html=True)


  




# 🔒 BLOCK SYSTEM IF NOT LOGGED IN
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ===============================
# EMAIL CONFIG
# ===============================
EMAIL_SENDER = "j.jaisurya2004@gmail.com"
EMAIL_PASSWORD = "xoolwrsxnqdiduxu"

PERSON_EMAILS = {
    "Jai": ["j.jaisurya2004@gmail.com"],
    "Unknown": ["j.jaisurya2004@gmail.com"]
}

def send_fine_email(person, fine):
    # If label exists in PERSON_EMAILS → use that
    if person in PERSON_EMAILS:
        receivers = PERSON_EMAILS[person]
    else:
        receivers = PERSON_EMAILS["Unknown"]

    for email in receivers:
        msg = EmailMessage()
        msg["Subject"] = "🚭 Smoking Fine Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = email
        msg.set_content(f"""
🚨 SMOKING DETECTED

👤 Person : {person}
💰 Fine   : ₹{fine}
⏰ Time   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)


# ===============================
# CONFIG
# ===============================
WORKSPACE = "vebbox"
WORKFLOW = "find-cigaretteunlits-pencils-and-cigarettelits"
FINE_AMOUNT = 500
CONF_THRESHOLD = 0.5
FRAME_SKIP = 6
DISTANCE_THRESHOLD = 90
COOLDOWN_SECONDS = 60

# ===============================
# ROBOFLOW CLIENT
# ===============================
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="0OprajKvjFAtWhjC0iEE"
)

# ===============================
# FACE MODELS
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()

# ===============================
# CLASS FORMAT
# ===============================
def format_class(cls):
    return {
        "pencil": "❌ Not Cigarette",
        "cigarette_unlit": "🚬 Cigarette (Unlit)",
        "cigarette_lit": "🔥 Smoking"
    }.get(cls, cls)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Control Panel")
if st.sidebar.button("▶ Start Detection"):
    st.session_state.run = True
if st.sidebar.button("⏹ Stop Detection"):
    st.session_state.run = False

st.sidebar.markdown("---")
if st.sidebar.button("↩ Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ===============================
# HEADER
# ===============================
st.markdown("""
<div style="text-align:center">
<h1>🚭 AI Smoking Detection System</h1>
<p style="font-size:20px;color:#22c55e">
Real-time detection • Face recognition • Auto fine
</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# LAYOUT
# ===============================
col1, col2 = st.columns([2.2, 1])
frame_box = col1.empty()
status_box = col2.empty()
objects_box = col2.empty()

# ===============================
# CAMERA LOOP
# ===============================
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    frame_id = 0
    detected_objects = []
    cigarette_points = []

    while st.session_state.run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if frame_id % FRAME_SKIP == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            img64 = base64.b64encode(buffer).decode()
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW,
                images={"image": f"data:image/jpeg;base64,{img64}"},
                use_cache=False
            )
            preds = result[0]["predictions"]["predictions"]
            detected_objects.clear()
            cigarette_points.clear()

            for p in preds:
                if p["confidence"] < CONF_THRESHOLD:
                    continue
                detected_objects.append((format_class(p["class"]), p["confidence"]))
                if p["class"] == "cigarette_lit":
                    cigarette_points.append((int(p["x"]), int(p["y"])))

        smoking = False
        person = "Unknown"

        for (x,y,w,h) in faces:
            mouth = (x+w//2, y+int(h*0.75))
            try:
                label, c = recognizer.predict(gray[y:y+h, x:x+w])
                if c < 70:
                    person = label_map[label].title()
            except:
                pass

            for cx, cy in cigarette_points:
                if np.linalg.norm(np.array(mouth)-np.array((cx,cy))) < DISTANCE_THRESHOLD:
                    smoking = True

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(frame,person,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        now = datetime.now()
        if smoking:
            status_box.markdown(f"""
            <div class="card danger">
            <h2>🔥 Smoking Detected</h2>
            <h3>{person}</h3>
            <h1>₹ {FINE_AMOUNT}</h1>
            </div>
            """, unsafe_allow_html=True)

            if not st.session_state.last_fine_time or \
               (now - st.session_state.last_fine_time).seconds > COOLDOWN_SECONDS:
                send_fine_email(person, FINE_AMOUNT)
                st.session_state.last_fine_time = now
        else:
            status_box.markdown("""
            <div class="card success">
            <h2>✅ No Smoking</h2>
            </div>
            """, unsafe_allow_html=True)

        objects_box.markdown("<div class='card'><h3>Detected Objects</h3></div>", unsafe_allow_html=True)
        for n,c in detected_objects:
            objects_box.title(f"- {n} — {c:.2f}")

        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=900)

    cap.release()
