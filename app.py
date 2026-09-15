"""
KidneyScan-AI — Backend (app.py)
=================================
Orchestrates two independently-trained models plus a rule-based lab
risk step. Nothing here retrains, replaces, or merges the two .pth
models — it wraps them correctly and runs them conditionally.

MODEL 1 (kidney_resnet18_bundle.pth): ResNet18 classifier
    -> Normal / Cyst / Stone / Tumor, confidence, activation-map
       bounding box ("red rectangle"). Reused from the existing app
       with ONE bug fix (see FIX NOTE below) — no other changes to
       its architecture, weights, or preprocessing.

MODEL 2 (best_kits19_unet (2).pth): MONAI residual U-Net
    -> tumor segmentation only, run ONLY when Model 1 predicts Tumor.
       Never used for classification; never used to compute volume
       from Model 1 or Grad-CAM output.

FIX NOTE (classification head):
    The checkpoint's trained head layers are stored under the key
    prefix "head." (head.1 / head.3 / head.5 / head.7), but the
    original app.py defined that same block as "self.fc", so
    load_state_dict(strict=False) silently skipped every head
    weight and the backbone ran with a randomly-initialized head.
    Fixed here by naming the module "self.head" to match the
    checkpoint exactly — same shapes, same layer order, zero change
    to weights or forward behavior, just makes the trained weights
    actually load.

UNVERIFIED ASSUMPTION (U-Net preprocessing):
    The checkpoint only tells us architecture (confirmed below), not
    how input slices were normalized during training. There is no
    training script, config file, or metadata in the checkpoint that
    records this. Per your instruction not to silently guess, this
    is called out explicitly:
        - Assumed here: resize to 256x256, single-channel grayscale,
          per-slice min-max normalization to [0, 1] (same convention
          the original app.py used for its (incorrect) U-Net).
        - This has NOT been verified against training code and may
          be wrong (e.g. if the model was trained on HU-windowed
          CT values instead of min-max-normalized ones). If
          segmentation quality looks off, this is the first thing to
          check — see PREPROCESS_UNET() below, which isolates the
          assumption to one function.

CONFIRMED U-Net architecture (parsed directly from the checkpoint's
pickle stream, key-for-key — see chat for the inspection method):
    - MONAI residual U-Net, spatial_dims=2 (all Conv2d kernels, 2D
      confirmed — this is a 2D, slice-wise model, not 3D)
    - in_channels=1 (grayscale slice)
    - out_channels=3  (background / kidney / tumor — NOT a single
      sigmoid channel; the original app.py's `sigmoid > 0.5` on one
      channel was wrong for this checkpoint and has been replaced
      with per-pixel argmax across 3 channels)
    - channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
      num_res_units=2 — inferred from the encoder/decoder tensor
      shapes; state_dict keys already match MONAI's default
      internal naming ("model.0...", "model.1...", "model.2..."),
      which is strong evidence this checkpoint was produced by
      monai.networks.nets.UNet with exactly this config.
"""

import os
import io
import tempfile
import numpy as np
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image, ImageDraw
from flask_sqlalchemy import SQLAlchemy
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from index import DASHBOARD_HTML

# ---------------------------------------------------------------------------
# App / DB setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///renal_diagnostics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'kidney_resnet18_bundle.pth')
UNET_PATH = os.path.join(MODELS_DIR, 'best_kits19_unet (2).pth')

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.nii', '.nii.gz'}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    creatinine = db.Column(db.Float)
    egfr = db.Column(db.Float)
    urine_protein = db.Column(db.Float)
    hematuria = db.Column(db.Boolean)
    risk_level = db.Column(db.String(50))
    ct_classification = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    lesion_volume_ml = db.Column(db.Float, nullable=True)
    lesion_diameter_mm = db.Column(db.Float, nullable=True)
    location = db.Column(db.String(100), nullable=True)
    bbox_coordinates = db.Column(db.String(100), nullable=True)

    def to_history_dict(self):
        return {
            "id": self.id,
            "classification": self.ct_classification,
            "confidence": self.confidence,
            "volume_ml": self.lesion_volume_ml,
            "diameter_mm": self.lesion_diameter_mm,
            "location": self.location,
        }


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# MODEL 1 — ResNet18 classifier (reused, one key-naming bug fixed)
# ---------------------------------------------------------------------------
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = base.avgpool
        # Named "head" (not "fc") to match the trained checkpoint's key
        # prefix exactly — see FIX NOTE at top of file. Layer shapes are
        # unchanged: 512 -> BN -> ReLU -> 256 -> BN -> ReLU -> 4.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return self.head(pooled)


CLASS_ORDER = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

classifier_model = ResNet18Classifier().to(device)

if os.path.exists(CLASSIFIER_PATH):
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    missing, unexpected = classifier_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARNING] Classifier load mismatch — missing={missing}, unexpected={unexpected}")
    else:
        print("[OK] ResNet18 classifier loaded (backbone + head).")
    # Prefer the checkpoint's own bundled metadata over hardcoded values,
    # since it's the source of truth for what the model was trained with.
    CLASS_ORDER = checkpoint.get('idx_to_class', CLASS_ORDER)
    NORM_MEAN = checkpoint.get('norm_mean', NORM_MEAN)
    NORM_STD = checkpoint.get('norm_std', NORM_STD)
    IMG_SIZE = checkpoint.get('img_size', IMG_SIZE)
else:
    print(f"[ERROR] Classifier checkpoint not found at {CLASSIFIER_PATH}")

classifier_model.eval()

CLF_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


# ---------------------------------------------------------------------------
# MODEL 2 — MONAI residual U-Net (segmentation only, Tumor-triggered)
# ---------------------------------------------------------------------------
try:
    from monai.networks.nets import UNet as MonaiUNet
    MONAI_AVAILABLE = True
except ImportError:
    MonaiUNet = None
    MONAI_AVAILABLE = False
    print("[ERROR] monai is not installed. Run: pip install monai")

UNET_INPUT_SIZE = 256  # matches original app.py's seg_transform; not separately confirmed

if MONAI_AVAILABLE:
    unet_model = MonaiUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,          # background / kidney / tumor
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    if os.path.exists(UNET_PATH):
        checkpoint = torch.load(UNET_PATH, map_location=device)
        state_dict = checkpoint.get('model_state', checkpoint)
        try:
            unet_model.load_state_dict(state_dict, strict=True)
            print("[OK] U-Net loaded with strict=True — architecture matches checkpoint exactly.")
        except RuntimeError as e:
            print(f"[WARNING] U-Net strict load failed, retrying non-strict: {e}")
            unet_model.load_state_dict(state_dict, strict=False)
        unet_model.eval()
    else:
        print(f"[ERROR] U-Net checkpoint not found at {UNET_PATH}")
else:
    unet_model = None

TUMOR_CHANNEL = 2  # background=0, kidney=1, tumor=2 (standard KiTS19 ordering)


def preprocess_unet_slice(slice_2d_uint8):
    """
    UNVERIFIED preprocessing (see module docstring). Isolated here on
    purpose: if segmentation output looks wrong, this is the function
    to revisit first — the architecture and weight-loading are solid,
    this normalization step is the one unconfirmed assumption.
    """
    img = Image.fromarray(slice_2d_uint8).convert('L').resize(
        (UNET_INPUT_SIZE, UNET_INPUT_SIZE)
    )
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def run_unet_on_slice(slice_2d_uint8):
    """Returns a boolean tumor mask at UNET_INPUT_SIZE resolution."""
    tensor = preprocess_unet_slice(slice_2d_uint8).to(device)
    with torch.no_grad():
        logits = unet_model(tensor)  # (1, 3, H, W)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return (pred == TUMOR_CHANNEL)


# ---------------------------------------------------------------------------
# File loading — PNG/JPG vs NIfTI are handled as genuinely different paths
# ---------------------------------------------------------------------------
def get_extension(filename):
    lower = filename.lower()
    if lower.endswith('.nii.gz'):
        return '.nii.gz'
    return os.path.splitext(lower)[1]


def load_2d_image(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert('RGB')


def load_nifti_volume(file_bytes, ext):
    import nibabel as nib
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        nii = nib.load(tmp_path)
        data = nii.get_fdata()
        affine = nii.affine
        spacing = nii.header.get_zooms()[:3]  # (sx, sy, sz) in mm
        return data, affine, spacing
    finally:
        os.remove(tmp_path)


def slice_to_uint8(slice_2d):
    slice_2d = np.nan_to_num(slice_2d)
    lo, hi = slice_2d.min(), slice_2d.max()
    return ((slice_2d - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# MODEL 1 inference: classification + activation-map bounding box
# (reused from the existing implementation — same 0.65 threshold, same
# color map, same left/right + upper/lower heuristic — not rewritten)
# ---------------------------------------------------------------------------
def classify_and_box(pil_img):
    tensor = CLF_TRANSFORM(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = classifier_model.features(tensor)
        pooled = classifier_model.avgpool(feat_map)
        logits = classifier_model.head(pooled)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_ORDER.get(pred_idx, CLASS_ORDER.get(str(pred_idx), "Unknown"))
        confidence = float(probs[pred_idx])
        all_probs = {CLASS_ORDER.get(i, CLASS_ORDER.get(str(i), str(i))): float(p)
                     for i, p in enumerate(probs)}

    img_copy = pil_img.copy().resize((256, 256))
    act_map = feat_map.squeeze(0).mean(dim=0).detach().cpu().numpy()
    act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
    thresh = act_map > 0.65
    y_idx, x_idx = np.where(thresh)
    w, h = img_copy.size

    if len(x_idx) > 0 and len(y_idx) > 0 and pred_class != 'Normal':
        fh, fw = act_map.shape
        x_min = int((x_idx.min() / fw) * w)
        x_max = int((x_idx.max() / fw) * w)
        y_min = int((y_idx.min() / fh) * h)
        y_max = int((y_idx.max() / fh) * h)
    else:
        x_min, y_min, x_max, y_max = 60, 60, 180, 180

    lat = "Right Kidney" if (x_min + x_max) / 2 > w / 2 else "Left Kidney"
    pole = "Upper Pole" if (y_min + y_max) / 2 < h / 2 else "Lower Pole"
    frame_location = f"{lat} / {pole}"

    draw = ImageDraw.Draw(img_copy)
    color_map = {'Tumor': '#ef4444', 'Stone': '#f97316', 'Cyst': '#eab308', 'Normal': '#22c55e'}
    box_color = color_map.get(pred_class, '#ef4444')
    if pred_class != 'Normal':
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=3)
        draw.text((x_min + 4, y_min + 4), pred_class, fill=box_color)

    buf = io.BytesIO()
    img_copy.save(buf, format="PNG")
    import base64
    annotated_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "label": pred_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "bounding_box": {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max},
        "frame_location": frame_location,
        "annotated_image": annotated_b64,
    }


# ---------------------------------------------------------------------------
# MODEL 2 pipeline + physical measurement engine
# ---------------------------------------------------------------------------
def segment_and_measure_nifti(volume, affine, spacing):
    """Runs U-Net slice-by-slice over a full CT volume, stacks the masks
    into a 3D segmentation, and computes physically-calibrated measurements
    using the NIfTI affine/spacing (never from voxel count alone)."""
    n_slices = volume.shape[2]
    mask_stack = np.zeros((UNET_INPUT_SIZE, UNET_INPUT_SIZE, n_slices), dtype=bool)
    scale_x = UNET_INPUT_SIZE / volume.shape[0]
    scale_y = UNET_INPUT_SIZE / volume.shape[1]

    for z in range(n_slices):
        slice_u8 = slice_to_uint8(volume[:, :, z])
        mask_stack[:, :, z] = run_unet_on_slice(slice_u8)

    voxel_count = int(mask_stack.sum())
    if voxel_count == 0:
        return {"performed": True, "tumor_found": False}

    # Effective spacing at the resized resolution
    eff_spacing = (spacing[0] / scale_x, spacing[1] / scale_y, spacing[2])
    voxel_volume_mm3 = eff_spacing[0] * eff_spacing[1] * eff_spacing[2]
    volume_ml = round((voxel_count * voxel_volume_mm3) / 1000.0, 2)

    components, n_components = ndi.label(mask_stack)

    coords_vox = np.argwhere(mask_stack)  # (N, 3) in resized-slice / original-z voxel space
    coords_phys = coords_vox * np.array(eff_spacing)  # approximate physical mm coords

    # Diameter via convex hull (avoids O(N^2) over millions of voxels):
    # take only the hull vertices of the point cloud, then the max
    # pairwise distance among that much smaller vertex set.
    diameter_mm = None
    dimensions_mm = None
    try:
        if len(coords_phys) >= 4:
            hull = ConvexHull(coords_phys)
            hull_pts = coords_phys[hull.vertices]
            max_d = 0.0
            for i in range(len(hull_pts)):
                d = np.linalg.norm(hull_pts[i + 1:] - hull_pts[i], axis=1)
                if len(d):
                    max_d = max(max_d, float(d.max()))
            diameter_mm = round(max_d, 2)
        extent = coords_phys.max(axis=0) - coords_phys.min(axis=0)
        dimensions_mm = [round(float(v), 2) for v in extent]
    except Exception as e:
        print(f"[WARNING] Geometry computation failed: {e}")

    centroid_vox = coords_vox.mean(axis=0)
    centroid_world = affine[:3, :3] @ np.array([
        centroid_vox[0] / scale_x, centroid_vox[1] / scale_y, centroid_vox[2]
    ]) + affine[:3, 3]

    location = determine_location_nifti(centroid_world, affine, volume.shape, spacing)

    return {
        "performed": True,
        "tumor_found": True,
        "components": int(n_components),
        "volume_ml": volume_ml,
        "maximum_diameter_mm": diameter_mm,
        "dimensions_mm": dimensions_mm,
        "centroid_world_mm": [round(float(c), 2) for c in centroid_world],
        "location": location,
        "voxel_count": voxel_count,
    }


def determine_location_nifti(centroid_world, affine, volume_shape, spacing):
    """Only returns a location when it can actually be derived from the
    NIfTI affine — never guessed."""
    try:
        import nibabel as nib
        axcodes = nib.aff2axcodes(affine)
        x_world = centroid_world[0]
        # RAS convention: +X = Right. If orientation isn't RAS-like on
        # the X axis we can't reliably call left/right.
        if 'R' in axcodes[0] or 'L' in axcodes[0]:
            side = "Right Kidney" if x_world > 0 else "Left Kidney"
        else:
            return "Location could not be reliably determined."

        z_extent_mm = volume_shape[2] * spacing[2]
        z_world = centroid_world[2]
        z_min = affine[2, 3]
        rel = (z_world - z_min) / (z_extent_mm + 1e-8)
        rel = min(max(rel, 0.0), 1.0)
        if rel < 0.33:
            vertical = "Lower Pole"
        elif rel < 0.66:
            vertical = "Mid Pole"
        else:
            vertical = "Upper Pole"
        return f"{side} / {vertical}"
    except Exception:
        return "Location could not be reliably determined."


def segment_and_measure_2d(pil_img):
    """Single 2D PNG/JPG: no physical calibration exists, so only
    pixel-domain measurements are returned — never mislabeled as mm/mL."""
    slice_u8 = np.array(pil_img.convert('L'))
    mask = run_unet_on_slice(slice_u8)
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return {"performed": True, "tumor_found": False}

    components, n_components = ndi.label(mask)
    ys, xs = np.where(mask)
    diameter_px = None
    try:
        pts = np.column_stack([xs, ys]).astype(float)
        if len(pts) >= 4:
            hull = ConvexHull(pts)
            hp = pts[hull.vertices]
            max_d = 0.0
            for i in range(len(hp)):
                d = np.linalg.norm(hp[i + 1:] - hp[i], axis=1)
                if len(d):
                    max_d = max(max_d, float(d.max()))
            diameter_px = round(max_d, 1)
    except Exception:
        pass

    return {
        "performed": True,
        "tumor_found": True,
        "components": int(n_components),
        "area_pixels": pixel_count,
        "maximum_diameter_pixels": diameter_px,
        "units_note": "No physical spacing available for a 2D image — "
                       "measurements are in pixels, not mm/mL.",
    }


# ---------------------------------------------------------------------------
# Lab risk module — rule-based, pluggable for a future trained model
# ---------------------------------------------------------------------------
def assess_lab_risk(creatinine, egfr, urine_protein, hematuria):
    """Triage-only. Never states a diagnosis — only whether further
    clinical evaluation / imaging may be appropriate. Swap this function
    body for a trained model's .predict() later without touching callers."""
    flags = []
    if egfr < 60:
        flags.append("reduced eGFR")
    if urine_protein > 30:
        flags.append("elevated urine protein")
    if hematuria:
        flags.append("hematuria present")
    if creatinine > 1.4:
        flags.append("elevated creatinine")

    if len(flags) >= 2:
        category = "Elevated"
    elif len(flags) == 1:
        category = "Moderate"
    else:
        category = "Low"

    recommend_ct = category in ("Elevated", "Moderate")
    message = (
        f"Risk category: {category}."
        + (f" Flagged factors: {', '.join(flags)}." if flags else " No flagged factors.")
        + (" Laboratory findings indicate that further clinical evaluation "
           "and imaging may be appropriate."
           if recommend_ct else " Values are within reference ranges.")
    )
    return category, recommend_ct, message


# ---------------------------------------------------------------------------
# Deterministic CT guidance — no generative AI or external API is used
# ---------------------------------------------------------------------------
def generate_tumor_guidance(
    tumor_diameter,
    tumor_area,
    location,
    confidence,
    connected_components,
    physical_spacing=None,
):
    """Build structured, rule-based guidance from existing model values."""
    diameter = float(tumor_diameter) if tumor_diameter is not None else None
    area = float(tumor_area) if tumor_area is not None else None
    confidence = float(confidence or 0.0)
    components = int(connected_components or 0)

    if physical_spacing is not None and diameter is not None:
        if diameter < 10:
            size_category = "Very small detected lesion"
        elif diameter <= 20:
            size_category = "Small detected lesion"
        elif diameter <= 40:
            size_category = "Moderate-size detected lesion"
        elif diameter <= 70:
            size_category = "Larger detected lesion"
        else:
            size_category = "Large detected lesion"
        size_message = (
            f"Maximum detected diameter: {diameter:.2f} mm. This is a "
            "screening category only and does not determine cancer or "
            "malignancy."
        )
        size_is_larger = diameter > 40
        size_is_smaller = diameter < 20
    elif diameter is not None:
        if diameter < 50:
            size_category = "Small image-region finding"
        elif diameter <= 150:
            size_category = "Moderate image-region finding"
        elif diameter <= 300:
            size_category = "Large image-region finding"
        else:
            size_category = "Very large image-region finding"
        size_message = (
            f"Maximum detected diameter: {diameter:.1f} px. Pixel measurements "
            "are image-relative and cannot be interpreted as physical tumor "
            "size without CT spacing/calibration."
        )
        size_is_larger = diameter > 150
        size_is_smaller = diameter < 50
    else:
        size_category = "Size could not be reliably measured"
        size_message = "No reliable diameter measurement was available."
        size_is_larger = False
        size_is_smaller = False

    if location and "Upper Pole" in location:
        location_message = (
            "Detected finding is located in the upper-pole region of the "
            "kidney on this image."
        )
    elif location and "Lower Pole" in location:
        location_message = (
            "Detected finding is located in the lower-pole region of the "
            "kidney on this image."
        )
    elif location and "Mid" in location:
        location_message = (
            "Detected finding is located in the mid-region of the kidney "
            "on this image."
        )
    else:
        location_message = (
            "Exact anatomical location could not be reliably determined "
            "from this image."
        )

    if confidence < 0.50:
        confidence_message = (
            "AI confidence is relatively low. The result should be treated "
            "cautiously and confirmed using the complete radiology study."
        )
    elif confidence < 0.75:
        confidence_message = "AI confidence is moderate. Clinical confirmation is recommended."
    else:
        confidence_message = (
            "AI confidence is higher, but the result still requires "
            "confirmation by a qualified clinician."
        )

    if components > 1:
        segmentation_message = (
            "Multiple segmented regions were detected. These should not "
            "automatically be interpreted as multiple tumors. Their clinical "
            "significance requires review of the complete CT study."
        )
    elif components == 1:
        segmentation_message = "One primary segmented region was detected in this image."
    else:
        segmentation_message = "No connected segmented region was reliably detected."

    precautions = [
        "Do not self-diagnose based on this AI result.",
        "Do not assume the detected lesion is cancer solely from the AI result.",
        "Do not start or stop medication based on this result.",
        "Do not make treatment decisions from this screening result alone.",
        "Review the complete CT examination and radiology report with a qualified clinician.",
    ]
    if size_is_larger:
        precautions.append("Larger detected findings warrant timely clinical evaluation.")
    if size_is_smaller:
        precautions.append(
            "Small size does not by itself determine whether a lesion is benign or malignant."
        )

    return {
        "tumor_size_assessment": {
            "category": size_category,
            "message": size_message,
            "area_pixels": area,
        },
        "location": location_message,
        "confidence": confidence_message,
        "segmentation": segmentation_message,
        "precautions": precautions,
        "next_steps": [
            "Review the complete CT study with a qualified radiologist.",
            "Discuss the finding with the appropriate clinician.",
            "Follow the clinician's recommendation for additional imaging or tests if required.",
        ],
        "important": (
            "This AI output is a screening/decision-support result and is not "
            "a diagnosis. It must not be used alone to make treatment decisions."
        ),
    }


def generate_ct_guidance(result: dict) -> dict:
    """Return structured guidance without changing any prediction values."""
    classification = result.get("classification", {})
    label = classification.get("label", "Unknown")
    if label == "Tumor":
        measurements = result.get("measurements", {})
        return generate_tumor_guidance(
            tumor_diameter=(
                measurements.get("maximum_diameter_mm")
                if measurements.get("maximum_diameter_mm") is not None
                else measurements.get("maximum_diameter_pixels")
            ),
            tumor_area=measurements.get("area_pixels"),
            location=measurements.get("location"),
            confidence=classification.get("confidence"),
            connected_components=result.get("segmentation", {}).get("components"),
            physical_spacing=result.get("physical_spacing"),
        )

    guidance_by_class = {
        "Stone": "Stone pattern detected. Arrange clinical review to assess the complete CT study.",
        "Cyst": "Cyst pattern detected. Ask a radiologist or clinician to confirm its features and follow-up needs.",
        "Normal": "No abnormality was identified by this classifier. This does not replace a formal radiology report.",
    }
    return {
        "summary": guidance_by_class.get(
            label,
            "The scan could not be assigned to a supported class. Request formal radiology review."
        ),
        "important": (
            "This AI output is a screening/decision-support result and is not "
            "a diagnosis. It must not be used alone to make treatment decisions."
        ),
    }


def get_diet_recommendation(prediction_class):
    """Return predefined general dietary information for a prediction class."""
    disclaimer = (
        "This is general dietary information, not a personalized medical diet. "
        "Consult a qualified doctor or dietitian for individualized advice."
    )
    general_recommendations = [
        "Choose balanced meals with vegetables, fruits, and whole grains.",
        "Maintain adequate hydration unless a qualified clinician has given you a fluid restriction.",
        "Limit excess salt and highly processed foods.",
    ]

    if prediction_class == "Normal":
        return {
            "title": "Balanced kidney-supportive diet",
            "recommendations": general_recommendations,
            "foods_to_include": ["Vegetables", "Fruits", "Whole grains", "Balanced protein sources"],
            "foods_to_limit": ["Highly processed foods", "Excess salt", "Sugar-sweetened drinks"],
            "note": disclaimer,
        }
    if prediction_class == "Cyst":
        return {
            "title": "General kidney-supportive diet for a cyst finding",
            "recommendations": [
                "Choose balanced foods and maintain adequate hydration unless restricted by a clinician.",
                "Use moderate salt and limit highly processed foods.",
                "Cysts do not have a specific food-based treatment.",
            ],
            "foods_to_include": ["Vegetables", "Fruits", "Whole grains", "Balanced protein sources"],
            "foods_to_limit": ["Highly processed foods", "Excess salt", "Foods high in added sugar"],
            "note": disclaimer,
        }
    if prediction_class == "Stone":
        return {
            "title": "General kidney-stone prevention guidance",
            "recommendations": [
                "Maintain adequate hydration unless a qualified clinician has given you a fluid restriction.",
                "Reduce excess salt and choose balanced calcium from food.",
                "Stone type was not determined, so specialized dietary restriction is not recommended from this result alone.",
            ],
            "foods_to_include": ["Water as clinically appropriate", "Balanced calcium-containing foods", "Vegetables and fruits"],
            "foods_to_limit": ["Excess salt", "Highly processed foods", "Large amounts of sugary drinks"],
            "note": disclaimer,
        }
    if prediction_class == "Tumor":
        return {
            "title": "Supportive nutrition for a tumor-pattern finding",
            "recommendations": [
                "Choose vegetables, fruits, whole grains, and appropriate protein sources.",
                "Maintain adequate hydration if allowed by your clinician.",
                "Diet does not treat or shrink a tumor; discuss nutrition needs with your care team.",
            ],
            "foods_to_include": ["Vegetables", "Fruits", "Whole grains", "Appropriate protein sources"],
            "foods_to_limit": ["Highly processed foods", "Excess salt", "Foods high in added sugar"],
            "note": disclaimer,
        }
    return {
        "title": "General healthy balanced diet",
        "recommendations": general_recommendations,
        "foods_to_include": ["Vegetables", "Fruits", "Whole grains", "Balanced protein sources"],
        "foods_to_limit": ["Highly processed foods", "Excess salt", "Sugar-sweetened drinks"],
        "note": disclaimer,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return DASHBOARD_HTML


@app.route('/api/assess-risk', methods=['POST'])
def api_assess_risk():
    data = request.json or {}
    creatinine = float(data.get('creatinine', 1.0))
    egfr = float(data.get('egfr', 90))
    urine_protein = float(data.get('urine_protein', 0))
    hematuria = bool(data.get('hematuria', False))

    category, recommend_ct, message = assess_lab_risk(creatinine, egfr, urine_protein, hematuria)

    record = PatientRecord(
        creatinine=creatinine, egfr=egfr, urine_protein=urine_protein,
        hematuria=hematuria, risk_level=category
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "record_id": record.id,
        "risk_category": category,
        "recommend_ct": recommend_ct,
        "message": message,
    })


@app.route('/api/analyze-ct', methods=['POST'])
def api_analyze_ct():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    file_bytes = file.read()
    record = PatientRecord.query.order_by(PatientRecord.id.desc()).first()

    try:
        if ext in ('.nii', '.nii.gz'):
            volume, affine, spacing = load_nifti_volume(file_bytes, ext)
            mid = volume.shape[2] // 2
            classify_img = Image.fromarray(slice_to_uint8(volume[:, :, mid])).convert('RGB')
        else:
            volume, affine, spacing = None, None, None
            classify_img = load_2d_image(file_bytes)

        classification = classify_and_box(classify_img)

        segmentation = {"performed": False}
        measurements = {}
        if classification["label"] == "Tumor":
            if volume is not None:
                seg_result = segment_and_measure_nifti(volume, affine, spacing)
            else:
                seg_result = segment_and_measure_2d(classify_img)
            segmentation = {
                "performed": seg_result.get("performed", False),
                "tumor_found": seg_result.get("tumor_found", False),
                "components": seg_result.get("components"),
            }
            measurements = {k: v for k, v in seg_result.items()
                             if k not in ("performed", "tumor_found", "components")}

    except Exception as e:
        print(f"[ERROR] CT analysis failed: {e}")
        return jsonify({"error": f"Failed to process scan: {str(e)}"}), 500

    result = {
        "classification": classification,
        "segmentation": segmentation,
        "measurements": measurements,
        "physical_spacing": list(spacing) if spacing is not None else None,
    }
    result["guidance"] = generate_ct_guidance(result)
    result["diet_recommendation"] = get_diet_recommendation(classification["label"])

    if record:
        record.ct_classification = classification["label"]
        record.confidence = classification["confidence"]
        record.lesion_volume_ml = measurements.get("volume_ml")
        record.lesion_diameter_mm = measurements.get("maximum_diameter_mm")
        record.location = measurements.get("location") or classification.get("frame_location")
        bb = classification["bounding_box"]
        record.bbox_coordinates = f"[{bb['xmin']}, {bb['ymin']}, {bb['xmax']}, {bb['ymax']}]"
        db.session.commit()

    return jsonify(result)


@app.route('/api/history', methods=['GET'])
def api_history():
    records = PatientRecord.query.order_by(PatientRecord.id.desc()).limit(20).all()
    return jsonify([r.to_history_dict() for r in records])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)