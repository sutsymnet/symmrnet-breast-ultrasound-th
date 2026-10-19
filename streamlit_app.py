"""
streamlit_app.py - SymMRNet-Sym2 breast ultrasound classification prototype
(bilingual display build: Thai default / English)
==========================================================================
Display-only fork of sutsymnet/symmrnet-breast-ultrasound @ v1.1-display.

Model, weights, preprocessing, decision rule and input gate are UNCHANGED:
  - symmrnet_core.py, input_gate.py and symmrnet_symlet2_3blocks.h5 are
    byte-identical copies from the original repo.
  - The only change is the UI text, switchable between Thai (default) and
    English. English mode reproduces v1.1-display wording, plus an English
    version of the Colour-Doppler warning.

Evaluated build referenced by the thesis remains v1.0-thesis
(commit 7a2cf8c2ee885014d0af1fa45af8820630730d28).

Requires in the same folder: symmrnet_core.py, input_gate.py,
symmrnet_symlet2_3blocks.h5
"""

import numpy as np
import streamlit as st
from PIL import Image

from symmrnet_core import (
    CLASS_NAMES,
    EXPECTED_PARAM_COUNT,
    MODEL_FILENAME,
    apply_symlet2_preprocessing_numpy,
    decode_and_resize,
    load_symmrnet,
)
from input_gate import check_image_bytes

APP_VERSION = "v1.2-display-th"
REPO_URL = "https://github.com/sutsymnet/symmrnet-breast-ultrasound-th"
EVAL_BUILD = "v1.0-thesis"
EVAL_COMMIT = "7a2cf8c2ee885014d0af1fa45af8820630730d28"

MODEL_LABEL = "SymMRNet-Symlet2-3Blocks (Wavelet-Sym2, preprocessed)"
TRAINING_SOURCE = "Kaggle: Ultrasound Breast Images for Breast Cancer (DS03.3)"
REPORTED_TEST_ACC = "93.90% (902 held-out images)"

# ------------------------------------------------------------------ UI text
T = {
    "th": {
        "page_title": "วิเคราะห์ภาพอัลตราซาวด์มะเร็งเต้านม",
        "about_h": "ℹ️ เกี่ยวกับ",
        "about_b": (
            "แอปต้นแบบนี้ใช้ **SymMRNet** ซึ่งเป็น CNN ที่ทุกขั้นการลดขนาด "
            "(downsampling) ใช้การพูลลิงด้วยเวฟเล็ตแบบคงที่ที่ได้จาก Symlet-2 "
            "เพื่อจำแนกภาพอัลตราซาวด์เต้านมเป็น **เนื้อดี (Benign)** หรือ "
            "**เนื้อร้าย (Malignant)**"
        ),
        "instr_h": "📋 วิธีใช้งาน",
        "instr_b": (
            "1. อัปโหลดภาพอัลตราซาวด์เต้านม\n"
            "2. กดปุ่ม **วิเคราะห์ภาพ**\n"
            "3. ดูผลการวิเคราะห์"
        ),
        "prov_h": "🔒 ที่มาของโมเดล",
        "disc_h": "⚠️ ข้อจำกัดความรับผิดชอบ",
        "disc_b": (
            "เครื่องมือนี้เป็นต้นแบบเพื่อการวิจัยเท่านั้น ไม่ใช่เครื่องมือแพทย์ "
            "และยังไม่ผ่านการรับรองสำหรับการใช้ทางคลินิก "
            "ควรปรึกษารังสีแพทย์ผู้เชี่ยวชาญทุกครั้ง"
        ),
        "title": "🩺 วิเคราะห์ภาพอัลตราซาวด์มะเร็งเต้านม — SymMRNet",
        "load_fail": "❌ โหลดโมเดลไม่สำเร็จ: {exc}",
        "load_fail_info": (
            "ไฟล์ `{f}` ต้องอยู่คู่กับ `streamlit_app.py` ที่ root ของ repo "
            "แอปนี้ตั้งใจไม่มีโมเดลสำรอง เพราะการรันโมเดลคนละตัวกับที่รายงาน"
            "ในวิทยานิพนธ์จะทำให้ผลลัพธ์ไม่ถูกต้อง"
        ),
        "load_ok": "✅ โหลด `{f}` แล้ว — ตรวจสอบพารามิเตอร์ {n:,} ตัว, input 64×64×1",
        "uploader": "เลือกภาพอัลตราซาวด์เต้านม...",
        "placeholder": "⬆️ กรุณาอัปโหลดภาพเพื่อเริ่มต้น",
        "gate_detail": "รายละเอียดทางเทคนิค",
        "gate_confirm": "ยืนยันว่าเป็นภาพ B-mode และต้องการดำเนินการต่อ",
        "gate_after": "⚠️ ผู้ใช้ยืนยันดำเนินการต่อ - ผลทำนายอาจคลาดเคลื่อน",
        "input_h": "ภาพที่อัปโหลด",
        "analyze": "🔍 วิเคราะห์ภาพ",
        "spinner": "กำลังวิเคราะห์ภาพด้วย SymMRNet…",
        "infer_fail": "การวิเคราะห์ล้มเหลว: {exc}",
        "result_h": "ผลการวิเคราะห์",
        "class": {"benign": "เนื้อดี (Benign)", "malignant": "เนื้อร้าย (Malignant)"},
        "score": "คะแนนจากโมเดล (ยังไม่ปรับเทียบ): {s:.4f}",
        "output_h": "**ผลลัพธ์ของโมเดลวิจัย**",
        "decision": (
            "เกณฑ์การตัดสิน: เลือกคลาสที่ค่า softmax สูงสุด (argmax) ซึ่งเทียบเท่า "
            "threshold 0.50 ตรงกับโปรโตคอลที่ใช้รายงานความแม่นยำชุดทดสอบ 93.90% "
            "ไม่มีการปรับ threshold รายภาพ"
        ),
        "ece": (
            "คะแนนนี้ยังไม่ผ่านการปรับเทียบ (ECE = 0.178 บนชุดทดสอบของสถาบัน, "
            "n = 236) จึงห้ามตีความเป็นความน่าจะเป็นในการวินิจฉัย "
            "เป็นต้นแบบเพื่อการวิจัย ไม่ใช่สำหรับใช้ทางคลินิก"
        ),
        "prep_h": "ขั้นตอนเตรียมภาพ (preprocessing) ที่ใช้กับภาพนี้",
        "prep_b": (
            "1. แปลงเป็นภาพระดับเทา (grayscale) ช่องเดียว\n"
            "2. ปรับขนาดเป็น 64×64 แบบ bilinear\n"
            "3. ปรับสเกลเป็นช่วง [0, 1]\n"
            "4. `pywt.dwt2(..., 'sym2', mode='symmetric')` ระดับเดียว\n"
            "5. ปรับสเกล sub-band: cA×1.05, cH×1.02, cV×1.02, cD×1.01\n"
            "6. สร้างภาพคืนด้วย `pywt.idwt2` ตัด/เติมขอบเป็น 64×64 "
            "และ clip ให้อยู่ใน [0, 1]\n\n"
            "ไม่ใช้ CLAHE ไม่ใช้ histogram equalisation และไม่มี augmentation "
            "— ตรงกับ pipeline ตอนเทรนทุกประการ"
        ),
        "footer": (
            "SymMRNet = Symlet Multi-Resolution Network การพูลลิงด้วยเวฟเล็ต"
            "ในที่นี้คือฟิลเตอร์ low-pass แบบ depthwise ที่ได้จาก Symlet-2 "
            "แบบคงที่ ตามด้วย ReLU และ average pooling ขนาด 2×2 — "
            "เป็นการประมาณ sub-band LL ไม่ใช่ DWT แบบ separable เต็มรูปแบบ"
        ),
        "source": "ซอร์สโค้ด",
    },
    "en": {
        "page_title": "SymMRNet Breast Ultrasound",
        "about_h": "ℹ️ About",
        "about_b": (
            "This prototype runs **SymMRNet**, a CNN in which every downsampling "
            "stage is a fixed Symlet-2-derived wavelet pooling operation, to "
            "classify breast ultrasound images as **Benign** or **Malignant**."
        ),
        "instr_h": "📋 Instructions",
        "instr_b": (
            "1. Upload a breast ultrasound image\n"
            "2. Click **Analyze Image**\n"
            "3. Review the classification result"
        ),
        "prov_h": "🔒 Model provenance",
        "disc_h": "⚠️ Disclaimer",
        "disc_b": (
            "Research prototype only. Not a medical device and not validated for "
            "clinical use. Always consult a qualified radiologist."
        ),
        "title": "🩺 Breast Ultrasound Classification - SymMRNet",
        "load_fail": "❌ Model failed to load: {exc}",
        "load_fail_info": (
            "`{f}` must sit next to `streamlit_app.py` in the repo "
            "root. There is deliberately no fallback model - serving a different "
            "network than the one reported in the thesis would invalidate the "
            "results."
        ),
        "load_ok": "✅ Loaded `{f}` - verified {n:,} parameters, input 64×64×1",
        "uploader": "Upload a breast ultrasound image",
        "placeholder": "⬆️ Upload an image to begin.",
        "gate_detail": "Technical detail",
        "gate_confirm": "I confirm this is a B-mode image and want to continue",
        "gate_after": "⚠️ User chose to continue - the prediction may be unreliable",
        "input_h": "Input",
        "analyze": "🔍 Analyze Image",
        "spinner": "Running SymMRNet…",
        "infer_fail": "Inference failed: {exc}",
        "result_h": "Result",
        "class": {"benign": "Benign", "malignant": "Malignant"},
        "score": "Model score (uncalibrated): {s:.4f}",
        "output_h": "**Research model output**",
        "decision": (
            "Decision rule: argmax over the softmax output (equivalent to a 0.50 "
            "threshold), identical to the protocol used for the reported "
            "93.90% test accuracy. No per-image threshold tuning is applied."
        ),
        "ece": (
            "Scores are uncalibrated (ECE = 0.178 on the institutional test set, "
            "n = 236) and must not be read as diagnostic probability. "
            "Research prototype - not for clinical use."
        ),
        "prep_h": "Preprocessing applied to this image",
        "prep_b": (
            "1. Decode as single-channel grayscale\n"
            "2. Bilinear resize to 64×64\n"
            "3. Scale to [0, 1]\n"
            "4. Single-level `pywt.dwt2(..., 'sym2', mode='symmetric')`\n"
            "5. Rescale sub-bands: cA×1.05, cH×1.02, cV×1.02, cD×1.01\n"
            "6. `pywt.idwt2` reconstruction, crop/pad to 64×64, clip to [0, 1]\n\n"
            "No CLAHE, no histogram equalisation, no augmentation - matching "
            "the training pipeline exactly."
        ),
        "footer": (
            "SymMRNet = Symlet Multi-Resolution Network. Wavelet pooling here is a "
            "fixed Symlet-2-derived low-pass depthwise filter followed by ReLU and "
            "2×2 average pooling - an approximation of the LL sub-band rather than a "
            "full separable DWT."
        ),
        "source": "Source",
    },
}

# English version of the Colour-Doppler warning. input_gate.py itself is left
# untouched (it returns Thai text); the app swaps in this text in English mode.
GATE_WARNING_EN = (
    "⚠️ **Colour signal detected — this may be a Color Doppler image**\n\n"
    "This system was trained and validated on **greyscale B-mode images** only. "
    "In a Color Doppler image, colour is overlaid on the lesion and replaces the "
    "real ultrasound echo information, so the prediction may be wrong in a "
    "direction that cannot be anticipated.\n\n"
    "If this is a normal B-mode image (e.g. it only has coloured calipers or an "
    "ROI box), you may confirm to continue."
)
GATE_ERRORS_EN = {
    "ภาพว่าง": "empty image",
    "ไม่พบพื้นที่ภาพ": "no image area found",
    "ภาพเล็กเกินไป": "image too small",
}


def gate_message_en(info):
    err = info.get("error") if isinstance(info, dict) else None
    if err is None:
        return GATE_WARNING_EN
    err = GATE_ERRORS_EN.get(err, err.replace("อ่านภาพไม่ได้: ", ""))
    return f"Could not read the image: {err}"


# --------------------------------------------------------- language & config
LANG_OPTIONS = {"th": "🇹🇭 ภาษาไทย", "en": "🇬🇧 English"}
if "lang" not in st.session_state:
    st.session_state.lang = "th"          # Thai is the default

st.set_page_config(page_title=T[st.session_state.lang]["page_title"],
                   page_icon="🩺", layout="wide")

st.radio(
    "Language / ภาษา",
    options=list(LANG_OPTIONS),
    format_func=LANG_OPTIONS.get,
    key="lang",
    horizontal=True,
    label_visibility="collapsed",
)
L = st.session_state.lang
t = T[L]


@st.cache_resource
def get_model():
    return load_symmrnet(MODEL_FILENAME, verify=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header(t["about_h"])
    st.write(t["about_b"])

    st.header(t["instr_h"])
    st.markdown(t["instr_b"])

    st.header(t["prov_h"])
    st.code(                                   # kept in English in both modes
        f"weights : {MODEL_FILENAME}\n"
        f"model   : {MODEL_LABEL}\n"
        f"input   : 64 x 64 x 1 (grayscale)\n"
        f"params  : {EXPECTED_PARAM_COUNT:,}\n"
        f"decision: argmax (threshold 0.50)\n"
        f"trained : {TRAINING_SOURCE}\n"
        f"test acc: {REPORTED_TEST_ACC}",
        language="text",
    )

    st.header(t["disc_h"])
    st.warning(t["disc_b"])


# --------------------------------------------------------------- main panel
st.title(t["title"])

try:
    model = get_model()
except Exception as exc:  # noqa: BLE001
    st.error(t["load_fail"].format(exc=exc))
    st.info(t["load_fail_info"].format(f=MODEL_FILENAME))
    st.stop()

st.success(t["load_ok"].format(f=MODEL_FILENAME, n=EXPECTED_PARAM_COUNT))

uploaded = st.file_uploader(
    t["uploader"],
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    key="uploader",
)

if uploaded is None:
    st.info(t["placeholder"])
    st.stop()

image_bytes = uploaded.getvalue()

needs_confirm, gate_msg, gate_info = check_image_bytes(image_bytes)
if needs_confirm:
    st.warning(gate_msg if L == "th" else gate_message_en(gate_info))
    with st.expander(t["gate_detail"]):
        st.json(gate_info)
    if not st.checkbox(t["gate_confirm"], key=f"confirm_{uploaded.name}"):
        st.stop()
    st.caption(t["gate_after"])

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(t["input_h"])
    st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)

if not st.button(t["analyze"], type="primary", use_container_width=True):
    st.stop()

with st.spinner(t["spinner"]):
    try:
        raw = decode_and_resize(image_bytes)                  # (64,64,1) in [0,255]
        batch = apply_symlet2_preprocessing_numpy(raw[None])  # (1,64,64,1) in [0,1]
        probs = model.predict(batch, verbose=0)[0]            # softmax, 2 classes
    except Exception as exc:  # noqa: BLE001
        st.error(t["infer_fail"].format(exc=exc))
        st.stop()

pred_idx = int(np.argmax(probs))
pred_label = t["class"][CLASS_NAMES[pred_idx]]
score = float(probs[pred_idx])

with col_right:
    st.subheader(t["result_h"])
    if pred_idx == 1:
        st.error(f"🔴 **{pred_label}**")
    else:
        st.success(f"🟢 **{pred_label}**")
    st.markdown(t["score"].format(s=score))

    st.write(t["output_h"])
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{t['class'][name]}: {p:.4f}")

    st.caption(t["decision"])
    st.caption(t["ece"])

    with st.expander(t["prep_h"]):
        st.markdown(t["prep_b"])

st.divider()
st.caption(t["footer"])

st.caption(
    f"SymMRNet-Sym2 · display build `{APP_VERSION}` · "
    f"evaluated build `{EVAL_BUILD}` "
    f"(commit `{EVAL_COMMIT}`) · "
    f"[{t['source']}]({REPO_URL})"
)
