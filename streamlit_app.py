"""Authenticated Streamlit UI for the existing KidneyScan-AI pipeline."""

import base64
import io
import os
import re
from datetime import datetime, timezone

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageEnhance

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


st.set_page_config(page_title="Kidney Sentry", page_icon="KS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#17212b; --muted:#687684; --teal:#087f8c; --mint:#e8f6f3; --line:#dce7e8; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
h1, h2, h3 { font-family:'Space Grotesk', sans-serif; }
.hero { padding:1.5rem 0 .8rem; border-bottom:1px solid var(--line); margin-bottom:1.2rem; }
.eyebrow { color:var(--teal); font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { margin:.2rem 0 .25rem; font-size:clamp(2rem,4vw,3.3rem); }
.hero p { color:var(--muted); max-width:690px; margin:0; }
.scan-frame img { max-height:520px; object-fit:contain; background:#f6faf9; border:1px solid var(--line); border-radius:12px; }
.result-band { background:var(--mint); border:1px solid #c9e9e3; border-radius:14px; padding:1.1rem 1.25rem; }
.muted { color:var(--muted); }
.status-card { border-radius:14px; border:1px solid var(--line); padding:1rem 1.2rem; margin:.5rem 0 1rem; }
.status-high { background:#fff0ef; border-color:#f1b8b3; }
.status-medium { background:#fff8e7; border-color:#efd28b; }
.status-low { background:#edf9f3; border-color:#b8e3ca; }
.status-label { font-weight:700; font-size:1.1rem; }
section[data-testid="stSidebar"] { border-right:1px solid var(--line); }
div[data-testid="stFileUploader"] { border:1px dashed #9dc9c5; border-radius:14px; padding:.35rem; }
</style>
""", unsafe_allow_html=True)


def secret(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def firebase_web_config():
    try:
        config = st.secrets.get("firebase", {})
        if config:
            return config
    except Exception:
        pass
    return {
        "api_key": secret("FIREBASE_API_KEY"),
        "project_id": secret("FIREBASE_PROJECT_ID"),
    }


def auth_request(endpoint, payload):
    api_key = firebase_web_config().get("api_key")
    if not api_key:
        raise RuntimeError("Firebase is not configured. Add firebase.api_key to Streamlit secrets.")
    response = requests.post(
        f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key}",
        json=payload,
        timeout=20,
    )
    if not response.ok:
        message = response.json().get("error", {}).get("message", "Authentication failed")
        raise RuntimeError(message.replace("EMAIL_EXISTS", "An account with this email already exists."))
    return response.json()


@st.cache_resource(show_spinner=False)
def firestore_client():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        try:
            firebase_admin.get_app()
        except ValueError:
            service_account = st.secrets.get("firebase_service_account")
            if service_account:
                firebase_admin.initialize_app(credentials.Certificate(dict(service_account)))
            else:
                path = secret("FIREBASE_SERVICE_ACCOUNT_PATH", "firebase_service_account.json")
                firebase_admin.initialize_app(credentials.Certificate(path))
        return firestore.client()
    except Exception as error:
        raise RuntimeError(f"Firestore is not configured: {error}") from error


def save_profile(uid, profile):
    client = firestore_client()
    data = dict(profile)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    client.collection("users").document(uid).set(data, merge=True)


def own_profile(uid):
    client = firestore_client()
    document = client.collection("users").document(uid).get()
    return document.to_dict() if document.exists else {}


def login_screen():
    st.markdown('<div class="hero"><div class="eyebrow">Kidney Sentry</div><h1>Private, practical scan review.</h1><p>Sign in to access your personal screening workspace and keep your profile data private.</p></div>', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            try:
                result = auth_request("signInWithPassword", {"email": email.strip(), "password": password, "returnSecureToken": True})
                st.session_state.update(auth_token=result["idToken"], user_id=result["localId"], user_email=result.get("email", email))
                try:
                    st.session_state.profile = own_profile(result["localId"])
                except Exception:
                    st.session_state.profile = {}
                st.rerun()
            except Exception as error:
                st.error(str(error))
    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Full name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", help="Use at least 6 characters.")
            height = st.number_input("Height (cm)", min_value=30.0, max_value=250.0, value=165.0)
            weight = st.number_input("Weight (kg)", min_value=2.0, max_value=300.0, value=65.0)
            gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Non-binary"])
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            if len(password) < 6:
                st.error("Password must contain at least 6 characters.")
            else:
                try:
                    result = auth_request("signUp", {"email": email.strip(), "password": password, "returnSecureToken": True})
                    profile = {"name": name.strip(), "email": result.get("email", email.strip()), "height": height, "weight": weight, "gender": gender}
                    save_profile(result["localId"], profile)
                    st.session_state.update(auth_token=result["idToken"], user_id=result["localId"], user_email=profile["email"], profile=profile)
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
    st.caption("Your password is handled by Firebase Authentication and is never written to Firestore.")


def render_list(title, values):
    st.markdown(f"**{title}**")
    for value in values:
        st.markdown(f"- {value}")


def image_from_value(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str) and value.startswith("data:image"):
        value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        elif array.ndim == 2:
            lo, hi = np.nanmin(array), np.nanmax(array)
            array = ((array - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    return Image.open(io.BytesIO(value)).convert("RGB")


def mask_visuals(mask, original):
    mask_array = np.asarray(mask, dtype=bool)
    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L").convert("RGB")
    base = original.convert("RGB").resize(mask_image.size)
    red = Image.new("RGB", mask_image.size, (214, 55, 55))
    overlay = Image.composite(red, base, ImageEnhance.Brightness(mask_image).enhance(0.55))
    return mask_image, overlay


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_scan(file_bytes, extension):
    if extension in (".nii", ".nii.gz"):
        volume, affine, spacing = load_nifti_volume(file_bytes, extension)
        display_image = Image.fromarray(slice_to_uint8(volume[:, :, volume.shape[2] // 2])).convert("RGB")
    else:
        display_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        volume, affine, spacing = None, None, None

    classification = classify_and_box(display_image)
    segmentation, measurements, mask_image, overlay_image = {"performed": False}, {}, None, None
    if classification["label"] == "Tumor":
        if not MONAI_AVAILABLE:
            st.warning("Segmentation is unavailable because MONAI is not installed.")
        elif volume is not None:
            segmented = segment_and_measure_nifti(volume, affine, spacing)
            middle_slice = slice_to_uint8(volume[:, :, volume.shape[2] // 2])
            mask = run_unet_on_slice(middle_slice)
            mask_image, overlay_image = mask_visuals(mask, display_image)
        else:
            segmented = segment_and_measure_2d(display_image)
            mask = run_unet_on_slice(np.array(display_image.convert("L")))
            mask_image, overlay_image = mask_visuals(mask, display_image)
        if MONAI_AVAILABLE:
            segmentation = {key: segmented.get(key) for key in ("performed", "tumor_found", "components")}
            measurements = {key: value for key, value in segmented.items() if key not in ("performed", "tumor_found", "components")}

    result = {"classification": classification, "segmentation": segmentation, "measurements": measurements, "physical_spacing": list(spacing) if spacing is not None else None}
    return display_image, classification, segmentation, measurements, mask_image, overlay_image, generate_ct_guidance(result), get_diet_recommendation(classification["label"])


def render_guidance(guidance):
    st.subheader("Guidance")
    if "tumor_size_assessment" in guidance:
        size = guidance["tumor_size_assessment"]
        st.markdown(f"**Tumor size assessment:** {size['category']}")
        st.caption(size["message"])
        st.write(f"**Location:** {guidance['location']}")
        st.write(f"**AI confidence:** {guidance['confidence']}")
        st.write(f"**Segmentation:** {guidance['segmentation']}")
        render_list("Precautions", guidance["precautions"])
        render_list("Recommended next steps", guidance["next_steps"])
    else:
        st.write(guidance["summary"])
    st.warning(guidance["important"])


def render_diet(diet):
    st.subheader("Diet recommendation")
    st.write(diet["title"])
    render_list("Foods to include", diet["foods_to_include"])
    render_list("Foods to limit", diet["foods_to_limit"])
    render_list("General recommendations", diet["recommendations"])
    st.warning(diet["note"])


def assess_ct_scan_necessity(urine_data):
    """Return conservative CT triage support from urine markers, not a diagnosis."""
    data = {key: float(value) if value not in (None, "") else None for key, value in urine_data.items() if key not in {"gross_blood", "flank_pain", "fever", "known_stone_history", "smoking_history"}}
    gross_blood = bool(urine_data.get("gross_blood"))
    flank_pain = bool(urine_data.get("flank_pain"))
    fever = bool(urine_data.get("fever"))
    known_stone_history = bool(urine_data.get("known_stone_history"))
    smoking_history = bool(urine_data.get("smoking_history"))

    protein = data.get("protein") or 0
    rbc = data.get("rbc") or 0
    leukocytes = data.get("leukocytes") or 0
    microalbumin = data.get("microalbumin") or 0
    abnormal = []
    if gross_blood or rbc >= 50:
        abnormal.append("marked or visible blood in urine")
    elif rbc >= 3:
        abnormal.append("microscopic hematuria")
    if protein >= 300:
        abnormal.append("severe proteinuria")
    elif protein >= 30:
        abnormal.append("proteinuria")
    if microalbumin >= 300:
        abnormal.append("severely increased microalbumin")
    elif microalbumin >= 30:
        abnormal.append("increased microalbumin")
    if leukocytes >= 10:
        abnormal.append("increased urinary leukocytes")
    abnormal_pH = data.get("ph") is not None and not 4.5 <= data["ph"] <= 8.0
    abnormal_sg = data.get("specific_gravity") is not None and not 1.003 <= data["specific_gravity"] <= 1.030
    if abnormal_pH:
        abnormal.append("pH outside the usual reference range")
    if abnormal_sg:
        abnormal.append("specific gravity outside the usual reference range")

    high_priority = gross_blood or rbc >= 50 or protein >= 300 or microalbumin >= 300
    high_priority = high_priority or ((rbc >= 3) and (protein >= 30 or microalbumin >= 30))
    high_priority = high_priority or (flank_pain and (rbc >= 3 or protein >= 30))
    risk_factors = [name for name, present in (("flank pain", flank_pain), ("fever", fever), ("known stone history", known_stone_history), ("smoking history", smoking_history)) if present]

    if high_priority:
        level = "Recommended"
        reasoning = "The reported findings include " + ", ".join(abnormal[:4]) + ". "
        if risk_factors:
            reasoning += "Together with " + ", ".join(risk_factors) + ", this warrants prompt clinician assessment to decide whether CT is appropriate."
        else:
            reasoning += "A clinician should review these findings promptly and decide whether CT or another imaging test is appropriate."
    elif abnormal or risk_factors:
        level = "Consider Consultation"
        reasoning = "The reported findings include " + ", ".join(abnormal[:4] or risk_factors) + ". Repeat testing, clinical examination, and consultation can determine whether imaging is needed."
    else:
        level = "Not Indicated at present"
        reasoning = "No strongly concerning urine marker pattern was entered. This does not rule out disease, and symptoms or persistent abnormalities still require clinical review."

    next_steps = [
        "Share the complete urine report and symptoms with a qualified clinician.",
        "Do not use this screening result to self-diagnose or delay urgent care.",
    ]
    if level == "Recommended":
        next_steps.insert(0, "Arrange prompt medical review to determine the appropriate imaging and laboratory follow-up.")
    else:
        next_steps.insert(0, "Consider repeat urinalysis and clinician review, especially if abnormalities persist.")
    precautions = ["CT choice, contrast use, and timing must be decided by a clinician based on kidney function, pregnancy status, symptoms, and examination."]
    if fever or flank_pain:
        precautions.append("Fever or significant flank pain with urinary changes may require same-day medical assessment.")
    return {"level": level, "reasoning": reasoning, "abnormal_markers": abnormal, "next_steps": next_steps, "precautions": precautions, "disclaimer": "AI decision-support only. This is not a diagnosis and does not replace a qualified clinician or radiology report."}


def extract_lab_text(uploaded_file):
    """Extract text when a supported parser is available; manual entry remains the source of truth."""
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
        except ImportError:
            return "PDF text extraction is unavailable. Enter the values manually."
    if name.endswith((".png", ".jpg", ".jpeg")):
        try:
            import importlib
            pytesseract = importlib.import_module("pytesseract")
            return pytesseract.image_to_string(Image.open(io.BytesIO(raw)))
        except (ImportError, OSError):
            return "Image OCR is unavailable. Enter the values manually."
    return ""


def parse_lab_value(text, labels):
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{label_pattern})[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def save_user_record(record_type, record):
    client = firestore_client()
    payload = dict(record)
    payload["record_type"] = record_type
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    client.collection("users").document(st.session_state.user_id).collection("history").add(payload)


def load_user_history():
    client = firestore_client()
    records = []
    for document in client.collection("users").document(st.session_state.user_id).collection("history").stream():
        records.append({"id": document.id, **document.to_dict()})
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)


def report_pdf(profile, result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import Image as PdfImage, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    display_image, classification, segmentation, measurements, mask_image, overlay_image, guidance, diet = result
    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    styles["Title"].alignment = TA_CENTER
    styles.add(styles["Normal"].clone("ReportSmall", fontSize=9, leading=12))
    styles.add(styles["Heading3"].clone("ReportHeading", spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#087f8c")))

    def safe(value, fallback="Not provided"):
        return str(value).strip() if value not in (None, "") else (fallback or "Not provided")

    def bullet_list(items):
        return ListFlowable(
            [ListItem(Paragraph(safe(item), styles["ReportSmall"]), leftIndent=12) for item in items],
            bulletType="bullet", start="circle", leftIndent=18,
        )

    # The email from Firebase Auth is authoritative even if the Firestore
    # profile was created before email was included in the profile document.
    profile = dict(profile or {})
    profile["email"] = safe(profile.get("email"), st.session_state.get("user_email"))
    story = [Paragraph("Kidney Sentry - Analysis Report", styles["Title"]), Paragraph("Screening support output; not a diagnosis.", styles["Normal"]), Spacer(1, 10)]
    profile_rows = [
        ["Patient name", safe(profile.get("name")), "Email", safe(profile.get("email"))],
        ["Height", f"{safe(profile.get('height'))} cm", "Weight", f"{safe(profile.get('weight'))} kg"],
        ["Gender", safe(profile.get("gender")), "Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    table = Table(profile_rows, colWidths=[2.2 * cm, 6 * cm, 2.2 * cm, 6 * cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3faf9")), ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#087f8c")), ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#087f8c")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#dce7e8")), ("PADDING", (0, 0), (-1, -1), 6)]))
    story += [table, Spacer(1, 14), Paragraph("Screening result", styles["Heading2"]), Paragraph(f"Prediction: <b>{safe(classification.get('label'))}</b> &nbsp;&nbsp; Confidence: <b>{classification.get('confidence', 0) * 100:.1f}%</b>", styles["Normal"])]
    image_items = [(display_image, "Original scan"), (image_from_value(classification["annotated_image"]), "Annotated result")]
    if overlay_image is not None:
        image_items += [(mask_image, "Segmentation mask"), (overlay_image, "Segmentation overlay")]
    for image, caption in image_items:
        image_buffer = io.BytesIO(); image.save(image_buffer, format="PNG"); image_buffer.seek(0)
        story += [Paragraph(caption, styles["Normal"]), PdfImage(image_buffer, width=8 * cm, height=8 * cm, kind="proportional"), Spacer(1, 6)]
    if measurements:
        story.append(Paragraph("Tumor measurements", styles["Heading2"]))
        measurement_rows = [[key.replace("_", " ").title(), safe(value)] for key, value in measurements.items()]
        measurement_table = Table(measurement_rows, colWidths=[7 * cm, 10 * cm])
        measurement_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#dce7e8")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3faf9")), ("PADDING", (0, 0), (-1, -1), 6)]))
        story.append(measurement_table)
    story += [Spacer(1, 8), Paragraph("Clinical guidance", styles["Heading2"])]
    if "tumor_size_assessment" in guidance:
        size = guidance["tumor_size_assessment"]
        story += [Paragraph(f"<b>Size assessment:</b> {safe(size.get('category'))}", styles["ReportSmall"]), Paragraph(safe(size.get("message")), styles["ReportSmall"]), Paragraph(f"<b>Location:</b> {safe(guidance.get('location'))}", styles["ReportSmall"]), Paragraph(f"<b>Confidence:</b> {safe(guidance.get('confidence'))}", styles["ReportSmall"]), Paragraph(f"<b>Segmentation:</b> {safe(guidance.get('segmentation'))}", styles["ReportSmall"]), Paragraph("<b>Precautions</b>", styles["ReportHeading"]), bullet_list(guidance.get("precautions", [])), Paragraph("<b>Recommended next steps</b>", styles["ReportHeading"]), bullet_list(guidance.get("next_steps", []))]
    else:
        story.append(Paragraph(safe(guidance.get("summary")), styles["ReportSmall"]))
    story += [Paragraph(f"<b>Important:</b> {safe(guidance.get('important'))}", styles["ReportSmall"]), Paragraph("Diet recommendation", styles["Heading2"]), Paragraph(f"<b>{safe(diet.get('title'))}</b>", styles["ReportSmall"]), Paragraph("Foods to include", styles["ReportHeading"]), bullet_list(diet.get("foods_to_include", [])), Paragraph("Foods to limit", styles["ReportHeading"]), bullet_list(diet.get("foods_to_limit", [])), Paragraph("General recommendations", styles["ReportHeading"]), bullet_list(diet.get("recommendations", [])), Paragraph(f"<b>Note:</b> {safe(diet.get('note'))}", styles["ReportSmall"])]
    document.build(story)
    return output.getvalue()


def render_ct_workspace(profile):
    st.subheader("CT scan and NIfTI analysis workspace")
    st.caption("Use the existing classifier, NIfTI slice handling, tumor segmentation, measurements, guidance, and PDF report pipeline.")
    uploaded_file = st.file_uploader("Upload a kidney image or CT volume", type=["png", "jpg", "jpeg", "nii", "gz"], help="NIfTI files may be uploaded as .nii or .nii.gz.", key="ct_upload")
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        extension = ".nii.gz" if filename.endswith(".nii.gz") else "." + filename.rsplit(".", 1)[-1]
        if extension not in ALLOWED_EXTENSIONS:
            st.error("Unsupported file type.")
        else:
            preview = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB") if extension not in (".nii", ".nii.gz") else None
            if preview is not None:
                st.image(preview, caption="Selected scan", use_container_width=True)
            if st.button("Analyze image", type="primary", use_container_width=True, key="analyze_ct"):
                try:
                    with st.spinner("Running the existing analysis pipeline..."):
                        st.session_state.analysis = analyze_scan(uploaded_file.getvalue(), extension)
                    classification = st.session_state.analysis[1]
                    measurements = st.session_state.analysis[3]
                    try:
                        save_user_record("ct_scan", {"filename": uploaded_file.name, "prediction": classification["label"], "confidence": classification["confidence"], "measurements": measurements})
                    except Exception as history_error:
                        st.warning(f"Analysis completed, but CT history could not be saved: {history_error}")
                    st.session_state.history_refresh = True
                except Exception as error:
                    st.error(f"Unable to analyze the uploaded scan: {error}")

    if "analysis" not in st.session_state:
        return
    result = st.session_state.analysis
    display_image, classification, segmentation, measurements, mask_image, overlay_image, guidance, diet = result
    st.divider()
    st.markdown('<div class="result-band">', unsafe_allow_html=True)
    result_col, confidence_col = st.columns(2)
    result_col.metric("Prediction", classification["label"])
    confidence_col.metric("Model confidence", f"{classification['confidence'] * 100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    image_col, annotated_col = st.columns(2)
    image_col.image(display_image, caption="Original image", use_container_width=True)
    annotated_col.image(image_from_value(classification["annotated_image"]), caption="AI output / annotated region", use_container_width=True)
    if classification["label"] == "Tumor":
        st.subheader("Tumor analysis")
        metric_values = {}
        if "area_pixels" in measurements: metric_values["Tumor area"] = f"{measurements['area_pixels']} px"
        if "maximum_diameter_mm" in measurements: metric_values["Maximum diameter"] = f"{measurements['maximum_diameter_mm']} mm"
        if "maximum_diameter_pixels" in measurements: metric_values["Maximum diameter"] = f"{measurements['maximum_diameter_pixels']} px"
        if segmentation.get("components") is not None: metric_values["Connected components"] = segmentation["components"]
        if measurements.get("location"): metric_values["Location"] = measurements["location"]
        metric_columns = st.columns(max(1, min(4, len(metric_values))))
        for index, (label, value) in enumerate(metric_values.items()): metric_columns[index % len(metric_columns)].metric(label, value)
        if measurements.get("units_note"): st.caption(measurements["units_note"])
        if mask_image is not None:
            mask_col, overlay_col = st.columns(2)
            mask_col.image(mask_image, caption="Tumor segmentation mask", use_container_width=True)
            overlay_col.image(overlay_image, caption="Segmentation overlay", use_container_width=True)
    render_guidance(guidance)
    render_diet(diet)
    try:
        st.download_button("Download PDF report", data=report_pdf(profile, result), file_name="kidney_sentry_report.pdf", mime="application/pdf", type="primary", use_container_width=True, key="download_ct_report")
    except Exception as error:
        st.warning(f"PDF report unavailable: {error}")


def render_urine_workspace():
    st.subheader("Urine lab analysis and CT recommendation")
    st.caption("Enter values from a urine report. Uploaded documents are used only to help prefill fields; verify every value against the original report.")
    report_file = st.file_uploader("Optional lab report (.txt, .pdf, .png, .jpg)", type=["txt", "pdf", "png", "jpg", "jpeg"], key="urine_report")
    extracted_text = ""
    if report_file is not None:
        extracted_text = extract_lab_text(report_file)
        if extracted_text and not extracted_text.startswith(("PDF text extraction", "Image OCR")):
            st.info("Report text extracted. Please verify the suggested values below.")
            with st.expander("View extracted text"):
                st.text(extracted_text[:5000])
    defaults = {
        "protein": parse_lab_value(extracted_text, ["protein", "urine protein"]) or 0.0,
        "rbc": parse_lab_value(extracted_text, ["rbc", "red blood cells", "blood"]) or 0.0,
        "leukocytes": parse_lab_value(extracted_text, ["leukocytes", "wbc", "white blood cells"]) or 0.0,
        "microalbumin": parse_lab_value(extracted_text, ["microalbumin", "albumin"]) or 0.0,
        "ph": parse_lab_value(extracted_text, ["ph"]) or 6.0,
        "specific_gravity": parse_lab_value(extracted_text, ["specific gravity", "sg"]) or 1.015,
    }
    with st.form("urine_analysis_form"):
        st.markdown("**Laboratory values**")
        value_col1, value_col2, value_col3 = st.columns(3)
        protein = value_col1.number_input("Protein (mg/dL)", min_value=0.0, value=float(defaults["protein"]), help="Use the units shown on the report.")
        rbc = value_col2.number_input("Blood / RBC (per HPF)", min_value=0.0, value=float(defaults["rbc"]))
        leukocytes = value_col3.number_input("Leukocytes (per HPF)", min_value=0.0, value=float(defaults["leukocytes"]))
        microalbumin = value_col1.number_input("Microalbumin (mg/g)", min_value=0.0, value=float(defaults["microalbumin"]))
        ph = value_col2.number_input("pH", min_value=0.0, max_value=14.0, value=float(defaults["ph"]))
        specific_gravity = value_col3.number_input("Specific gravity", min_value=1.0, max_value=1.1, value=float(defaults["specific_gravity"]), format="%.3f")
        st.markdown("**Symptoms and risk context**")
        symptom_col1, symptom_col2, symptom_col3 = st.columns(3)
        gross_blood = symptom_col1.checkbox("Visible blood")
        flank_pain = symptom_col2.checkbox("Flank pain")
        fever = symptom_col3.checkbox("Fever or chills")
        known_stone_history = symptom_col1.checkbox("Previous stone history")
        smoking_history = symptom_col2.checkbox("Smoking history")
        submitted = st.form_submit_button("Assess CT necessity", type="primary", use_container_width=True)
    if submitted:
        urine_data = {"protein": protein, "rbc": rbc, "leukocytes": leukocytes, "microalbumin": microalbumin, "ph": ph, "specific_gravity": specific_gravity, "gross_blood": gross_blood, "flank_pain": flank_pain, "fever": fever, "known_stone_history": known_stone_history, "smoking_history": smoking_history}
        assessment = assess_ct_scan_necessity(urine_data)
        st.session_state.urine_data = urine_data
        st.session_state.urine_assessment = assessment
        try:
            save_user_record("urine_analysis", {"urine_data": urine_data, "recommendation": assessment["level"], "reasoning": assessment["reasoning"]})
        except Exception as error:
            st.warning(f"Assessment completed, but history could not be saved: {error}")
    if "urine_assessment" in st.session_state:
        assessment = st.session_state.urine_assessment
        level_class = "status-high" if assessment["level"] == "Recommended" else "status-medium" if assessment["level"] == "Consider Consultation" else "status-low"
        st.markdown(f'<div class="status-card {level_class}"><div class="status-label">CT Scan Recommendation: {assessment["level"]}</div><div>{assessment["reasoning"]}</div></div>', unsafe_allow_html=True)
        st.markdown("**Recommended next steps**")
        render_list("", assessment["next_steps"])
        st.markdown("**Precautions**")
        render_list("", assessment["precautions"])
        st.warning(assessment["disclaimer"])


def render_profile_history():
    st.subheader("Patient profile and scan history")
    with st.form("profile_form"):
        name = st.text_input("Name", value=profile.get("name", ""))
        email = st.text_input("Email", value=profile.get("email", st.session_state.get("user_email", "")), disabled=True)
        profile_col1, profile_col2 = st.columns(2)
        height = profile_col1.number_input("Height (cm)", min_value=30.0, max_value=250.0, value=float(profile.get("height", 165.0)))
        weight = profile_col2.number_input("Weight (kg)", min_value=2.0, max_value=300.0, value=float(profile.get("weight", 65.0)))
        gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Non-binary"], index=["Prefer not to say", "Female", "Male", "Non-binary"].index(profile.get("gender", "Prefer not to say")) if profile.get("gender", "Prefer not to say") in ["Prefer not to say", "Female", "Male", "Non-binary"] else 0)
        if st.form_submit_button("Save profile", type="primary"):
            updated_profile = {"name": name.strip(), "email": email, "height": height, "weight": weight, "gender": gender}
            try:
                save_profile(st.session_state.user_id, updated_profile)
                st.session_state.profile = updated_profile
                st.success("Profile updated.")
            except Exception as error:
                st.error(f"Profile could not be saved: {error}")
    try:
        history = load_user_history()
        if history:
            st.markdown("**Your history**")
            for record in history:
                label = record.get("prediction") or record.get("recommendation", "Record")
                st.markdown(f"- **{label}** · {record.get('record_type', 'analysis').replace('_', ' ').title()} · {record.get('created_at', '')[:19].replace('T', ' ')}")
        else:
            st.info("No saved analyses yet.")
    except Exception as error:
        st.warning(f"History is unavailable: {error}")


if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "profile" not in st.session_state:
    st.session_state.profile = {}

if not st.session_state.auth_token:
    login_screen()
    st.stop()

profile = st.session_state.profile
with st.sidebar:
    st.markdown("### Kidney Sentry")
    st.caption(profile.get("name") or st.session_state.get("user_email", "Signed-in user"))
    if st.button("Log out", use_container_width=True):
        for key in ("auth_token", "user_id", "user_email", "profile", "analysis", "urine_data", "urine_assessment"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.caption("Screening and decision-support output only. Review results with a qualified clinician.")

st.markdown('<div class="hero"><div class="eyebrow">Personal screening workspace</div><h1>Kidney Sentry</h1><p>Review imaging and urine markers in one private, clinician-oriented workspace.</p></div>', unsafe_allow_html=True)
ct_tab, urine_tab, profile_tab = st.tabs(["🩺 CT Scan & NIfTI Analysis Workspace", "🧪 Urine Lab Analysis & CT Recommendation", "👤 Patient Profile & Scan History"])
with ct_tab:
    render_ct_workspace(profile)
with urine_tab:
    render_urine_workspace()
with profile_tab:
    render_profile_history()

with st.expander("Model files"):
    st.write(f"Classifier: {CLASSIFIER_PATH}")
    st.write(f"Tumor segmentation model: {UNET_PATH}")