"""
Kidney Disease Detection — FastAPI Backend v3
=============================================
ORIGINAL prediction logic (ResNet18 + Grad-CAM++) is 100% untouched.

NEW ADDITIONS:
  - Firebase Auth token verification (Bearer token on /predict)
  - Patient profile upsert  POST /patient
  - Firestore structure:
      patients/{uid}  ← profile
      patients/{uid}/history/{record_id}  ← per-scan records
  - Groq diet recommendation
  - Progress tracking  GET /patient/history
  - GET /health returns firebase/groq status
"""

import io, os, base64, traceback, json
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from streamlit import progress
import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision import transforms as T
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

# ── Firebase Admin ────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore

# ── Groq ─────────────────────────────────────────────────────────────────────
from groq import Groq

# =============================================================================
# FIREBASE INIT
# =============================================================================
_FB_SA_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "firebase_service_account.json")
try:
    cred = credentials.Certificate(_FB_SA_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_READY = True
    print("✅ Firebase Admin SDK initialised")
except Exception as _fe:
    FIREBASE_READY = False
    db = None
    print(f"⚠️  Firebase not initialised: {_fe}")

# =============================================================================
# GROQ INIT
# =============================================================================
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("✅ Groq ready" if groq_client else "⚠️  GROQ_API_KEY missing — using fallback diet text")

# =============================================================================
# CONFIG  (unchanged)
# =============================================================================
MODEL_PATH  = Path(__file__).parent / "models" / "kidney_resnet18_bundle.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
NORM_MEAN   = [0.485, 0.456, 0.406]
NORM_STD    = [0.229, 0.224, 0.225]
IMG_SIZE    = 224

# =============================================================================
# MODEL DEFINITION  (unchanged)
# =============================================================================
class KidneyResNet18(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn1  = nn.BatchNorm1d(512)
        self.fc1  = nn.Linear(512, 256)
        self.bn2  = nn.BatchNorm1d(256)
        self.fc2  = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn1(x); x = self.fc1(x)
        x = self.bn2(x); x = self.fc2(x)
        return x

# =============================================================================
# SMART LOADER  (unchanged)
# =============================================================================
def load_model():
    global NORM_MEAN, NORM_STD, IMG_SIZE
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        state        = payload["model_state"]
        idx_to_class = payload.get("idx_to_class", {i: c for i, c in enumerate(CLASS_NAMES)})
        classes      = [idx_to_class[i] for i in sorted(idx_to_class)]
        NORM_MEAN    = payload.get("norm_mean", NORM_MEAN)
        NORM_STD     = payload.get("norm_std",  NORM_STD)
        IMG_SIZE     = payload.get("img_size",  IMG_SIZE)
        last_w       = [v for k, v in state.items() if k.endswith(".weight") and v.dim() == 2]
        num_classes  = last_w[-1].shape[0] if last_w else len(classes)
        model        = KidneyResNet18(num_classes=num_classes)
        HEAD_REMAP   = {"head.1": "bn1", "head.3": "fc1", "head.5": "bn2", "head.7": "fc2"}
        remapped     = {}
        for k, v in state.items():
            new_k = k
            for old, new in HEAD_REMAP.items():
                if k.startswith(old + ".") or k == old:
                    new_k = new + k[len(old):]; break
            remapped[new_k] = v
        model.load_state_dict(remapped, strict=True)
        model.eval().to(DEVICE)
        return model, classes, NORM_MEAN, NORM_STD
    if isinstance(payload, dict):
        model = KidneyResNet18(num_classes=len(CLASS_NAMES))
        model.load_state_dict(payload, strict=False)
        model.eval().to(DEVICE)
        return model, CLASS_NAMES, NORM_MEAN, NORM_STD
    payload.eval().to(DEVICE)
    return payload, CLASS_NAMES, NORM_MEAN, NORM_STD

# =============================================================================
# GRAD-CAM++  (unchanged)
# =============================================================================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model; self.activations = None; self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self, m, i, o): self.activations = o.detach()
    def _save_gradient(self, m, gi, go): self.gradients   = go[0].detach()
    def generate(self, inp, target_class):
        self.model.zero_grad()
        out = self.model(inp)
        oh  = torch.zeros_like(out); oh[0, target_class] = 1.0
        out.backward(gradient=oh)
        grads = self.gradients[0]; acts = self.activations[0]
        g2, g3  = grads**2, grads**3
        sum_acts = acts.view(acts.shape[0], -1).sum(1).view(-1,1,1)
        eps   = 1e-8
        alpha = g2 / (2*g2 + sum_acts*g3 + eps)
        weights = (alpha * torch.relu(grads)).sum(dim=(1,2))
        cam = sum(w * acts[i] for i, w in enumerate(weights))
        cam = torch.relu(cam); cam -= cam.min(); cam /= cam.max() + eps
        return cam.cpu().numpy()

# =============================================================================
# BOUNDING BOX OVERLAY  (unchanged)
# =============================================================================
LABEL_COLORS = {"Normal":(0,200,0),"Cyst":(0,140,255),"Stone":(0,0,255),"Tumor":(0,0,255)}

def cam_to_bbox(image_np, cam, label, conf, percentile=92):
    H, W   = image_np.shape[:2]; result = image_np.copy()
    if label != "Normal":
        cam_r  = cv2.GaussianBlur(cv2.resize(cam,(W,H)), (7,7), 0)
        binary = (cam_r >= np.percentile(cam_r, percentile)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.putText(result, f"{label} ({conf*100:.1f}%)", (10,36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, LABEL_COLORS.get(label,(0,255,0)), 2)
    return result

# =============================================================================
# GROQ DIET RECOMMENDATION
# =============================================================================
_DIET_CONTEXT = {
    "Normal": "The kidney scan shows normal, healthy kidneys.",
    "Cyst":   "The kidney scan detected a kidney cyst (fluid-filled sac).",
    "Stone":  "The kidney scan detected kidney stones (hard mineral deposits).",
    "Tumor":  "The kidney scan detected a kidney tumor (abnormal tissue growth).",
}
_FALLBACK_DIET = {
    "Normal": "1. FOODS TO EAT\n• Plenty of water (2–3 L/day)\n• Fresh fruits and vegetables\n• Lean proteins\n• Whole grains\n• Low-fat dairy\n\n2. FOODS TO AVOID\n• Excessive sodium\n• Sugary beverages\n• Excess alcohol\n• Trans fats\n• Ultra-processed foods\n\n3. KEY LIFESTYLE TIPS\n• Maintain a healthy weight\n• Exercise 30 min/day\n• Annual kidney function tests\n\nConsult your nephrologist for personalised advice.",
    "Cyst":   "1. FOODS TO EAT\n• High-fibre foods (oats, legumes)\n• Antioxidant-rich berries\n• Leafy greens\n• Water (2–3 L/day)\n• Omega-3 rich fish\n\n2. FOODS TO AVOID\n• Excess sodium\n• Excess caffeine\n• Alcohol\n• High-fat red meat\n• Processed packaged foods\n\n3. KEY LIFESTYLE TIPS\n• Monitor blood pressure regularly\n• Maintain a healthy BMI\n• Follow-up imaging as advised\n\nConsult your nephrologist for personalised advice.",
    "Stone":  "1. FOODS TO EAT\n• Large amounts of water (3+ L/day)\n• Citrus fruits (lemon, orange)\n• Low-oxalate vegetables\n• Moderate dairy calcium\n• Magnesium-rich foods\n\n2. FOODS TO AVOID\n• High-oxalate foods (spinach, nuts, chocolate)\n• Excess salt\n• Excess animal protein\n• Sugary soft drinks\n• High-dose Vitamin C supplements\n\n3. KEY LIFESTYLE TIPS\n• Stay hydrated throughout the day\n• Aim for pale yellow urine\n• Reduce dietary sodium <2300 mg/day\n\nConsult your nephrologist for personalised advice.",
    "Tumor":  "1. FOODS TO EAT\n• Colourful fruits & vegetables\n• Cruciferous vegetables (broccoli, cauliflower)\n• Lean proteins (fish, poultry)\n• Whole grains\n• Green tea\n\n2. FOODS TO AVOID\n• Processed and red meats\n• Alcohol\n• High-sodium foods\n• Refined carbohydrates\n• Smoked or charred foods\n\n3. KEY LIFESTYLE TIPS\n• Follow your oncology team's dietary plan\n• Maintain adequate caloric intake\n• Report appetite/weight changes to your doctor\n\nConsult your nephrologist for personalised advice.",
}

def get_diet_recommendation(label: str) -> str:
    if not groq_client:
        return _FALLBACK_DIET.get(label, "Consult your nephrologist.")
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":(
                    "You are a clinical dietitian specialising in kidney health. "
                    "Respond with exactly 3 sections: "
                    "1. FOODS TO EAT (5 bullets), "
                    "2. FOODS TO AVOID (5 bullets), "
                    "3. KEY LIFESTYLE TIPS (3 bullets). "
                    "Each bullet is one concise sentence. "
                    "End with: 'Consult your nephrologist for personalised advice.'"
                )},
                {"role":"user","content":(
                    f"{_DIET_CONTEXT.get(label, label)}\n"
                    "Provide specific, actionable dietary recommendations."
                )},
            ],
            max_tokens=600, temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return _FALLBACK_DIET.get(label, "Consult your nephrologist.")

# =============================================================================
# PROGRESS TRACKING LOGIC
# =============================================================================
SEVERITY = {"Normal": 0, "Cyst": 1, "Stone": 2, "Tumor": 3}

def compute_progress(records: list[dict]) -> dict:
    """
    records: sorted oldest→newest list of {prediction, timestamp, ...}
    Returns:  {status, trend, summary, disease_counts, last_3}
    """
    if not records:
        return {"status": "No Data", "trend": "none", "summary": "No history yet.", "disease_counts": {}, "last_3": []}

    preds = [r["prediction"] for r in records]
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1

    last_3 = preds[-3:]
    scores  = [SEVERITY.get(p, 0) for p in last_3]

    if len(scores) == 1:
        trend  = "stable"
        status = "Stable"
    elif scores[-1] < scores[0]:
        trend  = "improving"
        status = "Improving ✅"
    elif scores[-1] > scores[0]:
        trend  = "worsening"
        status = "Critical ⚠️"
    else:
        trend  = "stable"
        status = "Stable 🔵"

    # Build human-readable summary
    arrow_map = {"improving": "↓ Getting better", "worsening": "↑ Getting worse", "stable": "→ Stable"}
    path = " → ".join(last_3)
    summary = f"{path}  ({arrow_map[trend]})"

    return {
        "status":        status,
        "trend":         trend,
        "summary":       summary,
        "disease_counts": counts,
        "last_3":        last_3,
        "total_scans":   len(records),
    }

# =============================================================================
# FIREBASE AUTH DEPENDENCY
# =============================================================================
_bearer = HTTPBearer(auto_error=False)

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    if not FIREBASE_READY:
        return {"uid": "anonymous", "email": "anonymous@local"}
    if creds is None:
        raise HTTPException(401, "Authorization header missing.")
    try:
        return fb_auth.verify_id_token(creds.credentials)
    except fb_auth.ExpiredIdTokenError:
        raise HTTPException(401, "Session expired. Please log in again.")
    except Exception as e:
        raise HTTPException(401, f"Authentication failed: {e}")

# =============================================================================
# APP INIT
# =============================================================================
app = FastAPI(title="KidneyScan AI", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

try:
    MODEL, CLASS_LIST, NORM_MEAN, NORM_STD = load_model()
    _target_layer = next(
        (l for l in reversed(list(MODEL.modules())) if isinstance(l, nn.Conv2d)), None
    )
    GRADCAM = GradCAMPlusPlus(MODEL, _target_layer)
    TRANSFORM = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    MODEL_LOADED = True
    print(f"✅ Model loaded on {DEVICE} | Classes: {CLASS_LIST}")
except Exception as e:
    MODEL_LOADED = False; LOAD_ERROR = str(e)
    print(f"❌ Model load error: {e}")

# =============================================================================
# SCHEMAS
# =============================================================================
class PatientProfile(BaseModel):
    name:     Optional[str]  = None
    age:      int
    mobile:   str
    diabetic: str            # "Yes" | "No" | "Unknown"

class PredictResponse(BaseModel):
    label:               str
    confidence:          float
    all_probs:           dict[str, float]
    result_image:        str
    diet_recommendation: str
    record_id:           Optional[str] = None

# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
def health():
    if not MODEL_LOADED:
        raise HTTPException(503, detail=f"Model not loaded: {LOAD_ERROR}")
    return {"status":"ok","device":str(DEVICE),"classes":CLASS_LIST,
            "firebase":FIREBASE_READY,"groq":groq_client is not None}

# ── Save / update patient profile ────────────────────────────────────────────
@app.post("/patient")
async def save_patient(profile: PatientProfile,
                       user: dict = Depends(get_current_user)):
    if not FIREBASE_READY:
        raise HTTPException(503, "Firebase not configured.")
    uid = user["uid"]
    doc_ref = db.collection("patients").document(uid)
    data = {
        "name":       profile.name or "",
        "age":        profile.age,
        "mobile":     profile.mobile,
        "diabetic":   profile.diabetic,
        "email":      user.get("email", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    doc = doc_ref.get()
    if not doc.exists:
        data["created_at"] = data["updated_at"]
    doc_ref.set(data, merge=True)
    return {"ok": True, "uid": uid}

# ── Get patient profile + history + progress ─────────────────────────────────
@app.get("/patient/history")
async def get_patient_history(user: dict = Depends(get_current_user)):
    try:
        if not FIREBASE_READY:
            raise HTTPException(503, "Firebase not configured.")

        uid = user["uid"]

        print("👉 Fetching history for UID:", uid)

        # Profile
        doc = db.collection("patients").document(uid).get()
        profile = doc.to_dict() if doc.exists else {}

        print("✅ Profile fetched")

        # History
        hist_docs = db.collection("patients").document(uid).collection("history").stream()

        records = []
        for d in hist_docs:
            data = d.to_dict()
            print("📄 Record:", data)
            records.append({"id": d.id, **data})

        print("✅ Total records:", len(records))

        # Safe sort
        records.sort(key=lambda x: str(x.get("timestamp", "")))

        progress = compute_progress(records)

        return {
            "profile": profile,
            "history": list(reversed(records)),
            "progress": progress
        }

    except Exception as e:
        print("🔥 FULL ERROR:", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

# ── PREDICT  (original logic untouched, just adds Firestore save) ─────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file:     UploadFile        = File(...),
    patient:  Optional[str]     = Form(None),   # JSON-encoded PatientProfile
    user:     dict              = Depends(get_current_user),
):
    if not MODEL_LOADED:
        raise HTTPException(503, f"Model not loaded: {LOAD_ERROR}")
    if file.content_type not in ("image/jpeg","image/png","image/jpg","image/webp"):
        raise HTTPException(400, "Only JPEG / PNG images are accepted.")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 20 MB).")

    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ── Original prediction logic (untouched) ──────────────────────────────
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(MODEL(tensor), dim=1).squeeze()
        conf, pred_idx = probs.max(dim=0)
        label      = CLASS_LIST[pred_idx.item()]
        confidence = conf.item()
        all_probs  = {CLASS_LIST[i]: round(p.item(),4) for i,p in enumerate(probs)}

        tensor_grad = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)
        cam         = GRADCAM.generate(tensor_grad, pred_idx.item())
        annotated   = cam_to_bbox(img_bgr, cam, label, confidence)
        _, buf      = cv2.imencode(".png", annotated)
        b64         = base64.b64encode(buf.tobytes()).decode()
        # ──────────────────────────────────────────────────────────────────────

        diet = get_diet_recommendation(label)

        # ── Save patient profile (if provided) + history record ───────────────
        record_id = None
        if FIREBASE_READY and db:
            uid = user["uid"]
            pat_ref = db.collection("patients").document(uid)

            # Upsert patient profile from form data
            if patient:
                try:
                    p = json.loads(patient)
                    profile_data = {
                        "name":       p.get("name",""),
                        "age":        p.get("age", 0),
                        "mobile":     p.get("mobile",""),
                        "diabetic":   p.get("diabetic","Unknown"),
                        "email":      user.get("email",""),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if not pat_ref.get().exists:
                        profile_data["created_at"] = profile_data["updated_at"]
                    pat_ref.set(profile_data, merge=True)
                except Exception as pe:
                    print(f"⚠️  Profile save failed: {pe}")

            # Save history record
            try:
                _, hist_ref = pat_ref.collection("history").add({
                    "prediction": label,
                    "confidence": round(confidence, 4),
                    "all_probs":  all_probs,
                    "diet":       diet,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                })
                record_id = hist_ref.id
            except Exception as he:
                print(f"⚠️  History save failed: {he}")

        return PredictResponse(
            label=label, confidence=round(confidence,4),
            all_probs=all_probs, result_image=b64,
            diet_recommendation=diet, record_id=record_id,
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, detail=traceback.format_exc())