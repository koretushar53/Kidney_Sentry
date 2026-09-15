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