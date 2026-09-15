<<<<<<< HEAD
"""Streamlit backup interface for the existing KidneyScan-AI pipeline."""

import io

import numpy as np
import streamlit as st
from PIL import Image

from app import (
    ALLOWED_EXTENSIONS,
    CLASSIFIER_PATH,
    MONAI_AVAILABLE,
    UNET_PATH,
    classify_and_box,
    generate_ct_guidance,
    get_diet_recommendation,
    load_nifti_volume,
    run_unet_on_slice,
    segment_and_measure_2d,
    segment_and_measure_nifti,
    slice_to_uint8,
)


st.set_page_config(page_title="KidneyScan-AI", page_icon=":medical_symbol:", layout="wide")

st.title("KidneyScan-AI")
st.caption("AI-Assisted Kidney Image Screening")
st.info(
    "Screening and decision-support output only. Results require confirmation "
    "by a qualified clinician."
)


def render_list(title, values):
    st.markdown(f"**{title}**")
    for value in values:
        st.markdown(f"- {value}")


def load_uploaded_scan(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    extension = ".nii.gz" if filename.endswith(".nii.gz") else filename.rsplit(".", 1)[-1]
    extension = extension if extension.startswith(".") else f".{extension}"

    if extension in (".nii", ".nii.gz"):
        volume, affine, spacing = load_nifti_volume(file_bytes, extension)
        middle_index = volume.shape[2] // 2
        display_image = Image.fromarray(
            slice_to_uint8(volume[:, :, middle_index])
        ).convert("RGB")
        return file_bytes, display_image, volume, affine, spacing

    display_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return file_bytes, display_image, None, None, None


def render_guidance(guidance):
    st.subheader("Guidance")
    if "tumor_size_assessment" in guidance:
        size = guidance["tumor_size_assessment"]
        st.markdown("**Tumor Size Assessment**")
        st.write(size["category"])
        st.caption(size["message"])
        st.markdown("**Location**")
        st.write(guidance["location"])
        st.markdown("**AI Confidence**")
        st.write(guidance["confidence"])
        st.markdown("**Segmentation**")
        st.write(guidance["segmentation"])
        render_list("Precautions", guidance["precautions"])
        render_list("Recommended Next Steps", guidance["next_steps"])
    else:
        st.write(guidance["summary"])
    st.warning(guidance["important"])


def render_diet(diet):
    st.subheader("Diet Recommendation")
    st.write(diet["title"])
    render_list("Foods to Include", diet["foods_to_include"])
    render_list("Foods to Limit", diet["foods_to_limit"])
    render_list("General Recommendations", diet["recommendations"])
    st.warning(diet["note"])


uploaded_file = st.file_uploader(
    "Upload a kidney image or CT volume",
    type=["png", "jpg", "jpeg", "nii", "gz"],
    help="NIfTI files may be uploaded as .nii or .nii.gz.",
)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    extension = ".nii.gz" if filename.endswith(".nii.gz") else "." + filename.rsplit(".", 1)[-1]
    if extension not in ALLOWED_EXTENSIONS:
        st.error("Unsupported file type.")
    elif not st.button("Analyze Image", type="primary"):
        st.image(uploaded_file, caption="Original image", use_container_width=True)
    else:
        try:
            _, display_image, volume, affine, spacing = load_uploaded_scan(uploaded_file)
            classification = classify_and_box(display_image)

            segmentation = {"performed": False}
            measurements = {}
            mask_image = None
            if classification["label"] == "Tumor":
                if not MONAI_AVAILABLE:
                    st.warning("Segmentation is unavailable because MONAI is not installed.")
                elif volume is not None:
                    segmented = segment_and_measure_nifti(volume, affine, spacing)
                    segmentation = {
                        "performed": segmented.get("performed", False),
                        "tumor_found": segmented.get("tumor_found", False),
                        "components": segmented.get("components"),
                    }
                    measurements = {
                        key: value for key, value in segmented.items()
                        if key not in ("performed", "tumor_found", "components")
                    }
                    middle_slice = slice_to_uint8(volume[:, :, volume.shape[2] // 2])
                    mask_image = run_unet_on_slice(middle_slice)
                else:
                    segmented = segment_and_measure_2d(display_image)
                    segmentation = {
                        "performed": segmented.get("performed", False),
                        "tumor_found": segmented.get("tumor_found", False),
                        "components": segmented.get("components"),
                    }
                    measurements = {
                        key: value for key, value in segmented.items()
                        if key not in ("performed", "tumor_found", "components")
                    }
                    mask_image = run_unet_on_slice(np.array(display_image.convert("L")))

            result = {
                "classification": classification,
                "segmentation": segmentation,
                "measurements": measurements,
                "physical_spacing": list(spacing) if spacing is not None else None,
            }
            guidance = generate_ct_guidance(result)
            diet = get_diet_recommendation(classification["label"])

            st.divider()
            st.subheader("Prediction Result")
            result_col, confidence_col = st.columns(2)
            result_col.metric("Prediction", classification["label"])
            confidence_col.metric("Model confidence", f"{classification['confidence'] * 100:.1f}%")

            image_col, annotated_col = st.columns(2)
            image_col.image(display_image, caption="Original image", use_container_width=True)
            annotated_col.image(
                classification["annotated_image"],
                caption="AI output / annotated region",
                use_container_width=True,
            )

            if classification["label"] == "Tumor":
                st.subheader("Tumor Analysis")
                metric_values = {}
                if "area_pixels" in measurements:
                    metric_values["Tumor area"] = f"{measurements['area_pixels']} px"
                if "maximum_diameter_mm" in measurements:
                    metric_values["Maximum diameter"] = f"{measurements['maximum_diameter_mm']} mm"
                if "maximum_diameter_pixels" in measurements:
                    metric_values["Maximum diameter"] = f"{measurements['maximum_diameter_pixels']} px"
                if segmentation.get("components") is not None:
                    metric_values["Connected components"] = segmentation["components"]
                if measurements.get("location"):
                    metric_values["Location"] = measurements["location"]
                metric_columns = st.columns(max(1, min(4, len(metric_values))))
                for index, (label, value) in enumerate(metric_values.items()):
                    metric_columns[index % len(metric_columns)].metric(label, value)
                if measurements.get("units_note"):
                    st.caption(measurements["units_note"])
                if mask_image is not None:
                    st.image(mask_image, caption="Tumor segmentation mask", clamp=True, use_container_width=True)

            render_guidance(guidance)
            render_diet(diet)
        except Exception as error:
            st.error(f"Unable to analyze the uploaded scan: {error}")

with st.expander("Model files"):
    st.write(f"Classifier: {CLASSIFIER_PATH}")
    st.write(f"Tumor segmentation model: {UNET_PATH}")
=======
"""
KidneyScan AI — Streamlit Frontend
====================================
Fixed version with:
  - Firebase Email/Password Authentication
  - Prediction via FastAPI backend
  - Diet recommendation display
  - Patient history display
"""

import os
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Your Firebase Web API Key (from Firebase Console → Project Settings → General)
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "AIzaSyAeR3hCK3cENO5yeIjkXh0sr5TqjY8IJMg")

FIREBASE_SIGN_IN_URL  = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
FIREBASE_SIGN_UP_URL  = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"

CLASS_COLORS = {
    "Normal": "#00e676",
    "Cyst":   "#64b5f6",
    "Stone":  "#ffab40",
    "Tumor":  "#ef5350",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KidneyScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #080c14; }
  [data-testid="stSidebar"]          { background: #0e1520; }
  h1, h2, h3                         { color: #e8edf5; }
  .stButton > button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white; border: none; border-radius: 8px;
    font-weight: 600;
  }
  .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("id_token",   None),
    ("user_email", None),
    ("auth_mode",  "login"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# FIREBASE AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def firebase_sign_in(email: str, password: str) -> dict:
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(FIREBASE_SIGN_IN_URL, json=payload, timeout=10)
    return r.json()


def firebase_sign_up(email: str, password: str) -> dict:
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(FIREBASE_SIGN_UP_URL, json=payload, timeout=10)
    return r.json()


def friendly_firebase_error(err: dict) -> str:
    code = err.get("error", {}).get("message", "UNKNOWN")
    map_ = {
        "EMAIL_NOT_FOUND":          "No account found with this email.",
        "INVALID_PASSWORD":         "Incorrect password.",
        "EMAIL_EXISTS":             "Email is already registered. Please sign in.",
        "WEAK_PASSWORD":            "Password must be at least 6 characters.",
        "INVALID_EMAIL":            "Please enter a valid email address.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please wait and try again.",
        "INVALID_LOGIN_CREDENTIALS":"Invalid email or password.",
    }
    return map_.get(code, f"Auth error: {code}")


# ─────────────────────────────────────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def show_auth_screen():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("## 🔬 KidneyScan AI")
        st.markdown("##### Sign in to access the diagnostic tool")
        st.divider()

        tabs = st.tabs(["Sign In", "Register"])

        # ── Sign In ──────────────────────────────────────────────────────────
        with tabs[0]:
            email    = st.text_input("Email address", key="login_email", placeholder="doctor@hospital.com")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            if st.button("Sign In", use_container_width=True, key="login_btn"):
                if not email or not password:
                    st.error("Please enter your email and password.")
                else:
                    with st.spinner("Signing in…"):
                        result = firebase_sign_in(email, password)
                    if "idToken" in result:
                        st.session_state.id_token   = result["idToken"]
                        st.session_state.user_email = result.get("email", email)
                        st.rerun()
                    else:
                        st.error(friendly_firebase_error(result))

        # ── Register ─────────────────────────────────────────────────────────
        with tabs[1]:
            reg_email = st.text_input("Email address", key="reg_email", placeholder="doctor@hospital.com")
            reg_pass  = st.text_input("Password (min 6 chars)", type="password", key="reg_pass")
            if st.button("Create Account", use_container_width=True, key="reg_btn"):
                if not reg_email or not reg_pass:
                    st.error("Please fill in all fields.")
                else:
                    with st.spinner("Creating account…"):
                        result = firebase_sign_up(reg_email, reg_pass)
                    if "idToken" in result:
                        st.session_state.id_token   = result["idToken"]
                        st.session_state.user_email = result.get("email", reg_email)
                        st.success("Account created! You are now signed in.")
                        st.rerun()
                    else:
                        st.error(friendly_firebase_error(result))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def show_main_app():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔬 KidneyScan AI")
        st.markdown(f"**Signed in as:**  \n`{st.session_state.user_email}`")
        st.divider()

        # Health check
        try:
            health = requests.get(f"{API_BASE}/health", timeout=5).json()
            st.success(f"✅ Model Online · {health.get('device', 'cpu').upper()}")
        except Exception:
            st.error("❌ Backend offline — start uvicorn main:app --reload")

        st.divider()
        if st.button("Sign Out", use_container_width=True):
            st.session_state.id_token   = None
            st.session_state.user_email = None
            st.rerun()

        st.caption("For research / demonstration only.")

    # ── Main content ─────────────────────────────────────────────────────────
    st.title("🔬 Kidney Disease Detection")
    st.markdown("Upload a kidney CT scan image to get an AI-powered diagnosis with diet recommendations.")
    st.divider()

    col_left, col_right = st.columns([1, 1.3], gap="large")

    # ── LEFT: Upload ──────────────────────────────────────────────────────────
    with col_left:
        st.subheader("📤 Upload CT Scan")
        uploaded = st.file_uploader(
            "Choose a kidney CT scan image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.image(uploaded, caption="Uploaded CT Scan", use_column_width=True)
            st.caption(f"📄 {uploaded.name}  ·  {uploaded.size / 1024:.1f} KB")

            if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
                with st.spinner("Running analysis… this may take a few seconds"):
                    try:
                        headers = {"Authorization": f"Bearer {st.session_state.id_token}"}
                        files   = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                        res     = requests.post(f"{API_BASE}/predict", headers=headers, files=files, timeout=60)

                        if res.status_code == 401:
                            st.error("Session expired. Please sign out and sign in again.")
                        elif res.ok:
                            st.session_state["last_result"] = res.json()
                        else:
                            detail = res.json().get("detail", "Unknown error")
                            st.error(f"Backend error: {detail}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Is uvicorn running?")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── RIGHT: Results ────────────────────────────────────────────────────────
    with col_right:
        st.subheader("📊 Analysis Results")

        result = st.session_state.get("last_result")
        if not result:
            st.info("Upload a CT scan and click **Analyze Image** to see the results.")
        else:
            label      = result.get("label", "—")
            confidence = result.get("confidence", 0)
            all_probs  = result.get("all_probs", {})
            diet       = result.get("diet_recommendation", "")
            b64_img    = result.get("result_image", "")

            # Diagnosis card
            color = CLASS_COLORS.get(label, "#00c6ff")
            st.markdown(
                f"""<div style="background:rgba(0,0,0,0.3); border:1px solid {color}44;
                     border-radius:12px; padding:20px; margin-bottom:16px;">
                  <div style="font-size:11px; letter-spacing:1px; color:#6b7a96; margin-bottom:6px;">DIAGNOSIS</div>
                  <div style="display:flex; align-items:baseline; justify-content:space-between;">
                    <span style="font-size:32px; font-weight:700; color:{color};">{label}</span>
                    <span style="font-size:22px; color:#6b7a96;">{confidence*100:.1f}%</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Class probabilities
            st.markdown("**Class Probabilities**")
            for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                c = CLASS_COLORS.get(cls, "#fff")
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
                    f'<span style="width:60px; font-size:13px; color:{c};">{cls}</span>'
                    f'<div style="flex:1; background:#141e2e; border-radius:4px; height:8px;">'
                    f'<div style="width:{prob*100:.1f}%; background:{c}; height:100%; border-radius:4px;"></div></div>'
                    f'<span style="width:42px; text-align:right; font-size:12px; color:#6b7a96;">{prob*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Annotated image
            if b64_img:
                st.markdown("**Grad-CAM++ Localization**")
                import base64
                img_bytes = base64.b64decode(b64_img)
                st.image(img_bytes, caption="Red box = anomaly region detected by Grad-CAM++", use_column_width=True)

    # ── Diet Recommendation (full width below) ────────────────────────────────
    result = st.session_state.get("last_result")
    if result and result.get("diet_recommendation"):
        st.divider()
        st.subheader("🥗 Diet & Lifestyle Recommendation")
        diet = result["diet_recommendation"]
        label = result.get("label", "")
        color = CLASS_COLORS.get(label, "#00c6ff")

        st.markdown(
            f'<div style="background:rgba(0,198,255,0.04); border:1px solid rgba(0,198,255,0.15); '
            f'border-radius:12px; padding:24px;">'
            f'<pre style="font-family: monospace; color:#e8edf5; white-space:pre-wrap; '
            f'font-size:13px; line-height:1.75; margin:0;">{diet}</pre>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("⚠️ This recommendation is AI-generated for informational purposes only. Always consult a qualified nephrologist.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.id_token is None:
    show_auth_screen()
else:
    show_main_app()
>>>>>>> b01f96785d348604e61c39ff410e26a5d0477b0b
