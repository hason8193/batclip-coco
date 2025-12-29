"""
Streamlit demo that lets you classify local images with BATCLIP-style TTA.

Users can enter arbitrary topic labels, upload photos, and optionally run
unsupervised test-time adaptation (entropy minimization over the uploaded batch)
via prompt tuning. It uses the same CLIP+soft-prompt stack that BATCLIP exposes
so the adaptation behaves similarly to the published method.

Run with:

    streamlit run demo_app.py

Prerequisites:

    pip install streamlit open-clip-torch

The demo works on CPU as well but will adapt faster on GPU.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import json

import streamlit as st
import torch
from PIL import Image

# Optional: Gemini caption generator for class prompts
try:
    from google import genai as genai_client
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Prefer env key (avoid hard-coding secrets in repo)
DEFAULT_GENAI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Prefer using the shared generator module if present (keeps demo + script consistent)
try:
    from LLM_Caption.generate_main import generate_captions as shared_generate_captions  # type: ignore
    HAVE_SHARED_CAPTION_GEN = True
except Exception:
    HAVE_SHARED_CAPTION_GEN = False
    shared_generate_captions = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = ROOT_DIR / "classification"
if CLASSIFICATION_DIR.exists():
    sys.path.append(str(CLASSIFICATION_DIR))

IMPORT_ERROR: Optional[BaseException] = None
try:
    from open_clip import create_model_and_transforms
    from models.custom_clip import PromptLearner, TextEncoder
    from utils.losses import Entropy
except ImportError as exc:  # pragma: no cover - runtime import guard
    IMPORT_ERROR = exc


def _extract_sentences(response_text: str) -> List[str]:
    return [line.strip() for line in response_text.split("\n") if line.strip()]


def _generate_llm_captions_inline(
    labels: Sequence[str],
    api_key: str,
    out_path: Path,
    temperature: float = 0.6,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """
    Generate short descriptive captions per class using Google Gemini.
    Returns a dict {class_name: [sentences]} and also merges into out_path JSON.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai is not installed.")
    if not labels:
        return {}

    # load existing
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    client = genai_client.Client(api_key=api_key)
    prompt_header = (
        "You're an expert writing objective and general descriptions for things in English. "
        "Write short sentence less than 40 words to answer each of these questions in english. "
        "Exactly 10 sentences for each question. 1 sentence on 1 line. "
        "The english name of the main subject must appear in every sentence. "
        "Translate the name to english. Do not reiterate the questions. Do not provide sources."
    )

    new_entries = {}
    for cls in labels:
        cls = cls.strip()
        if not cls or cls in existing:
            continue
        class_prompt = (
            f"{prompt_header}\n"
            f"Describe what a {cls} looks like.\n"
            f"How can you identify a {cls} by appearances?\n"
            f"What does a {cls} look like?\n"
            f"Describe an image from the internet of a {cls}.\n"
            f"A caption of an image of a {cls}:"
        )

        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=class_prompt)],
            )
        ]
        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                tools=[genai_types.Tool(googleSearch=genai_types.GoogleSearch())],
            ),
        )
        sentences = _extract_sentences(resp.text)
        if sentences:
            existing[cls] = sentences
            new_entries[cls] = sentences

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return new_entries


if IMPORT_ERROR is None:
    class BATCLIPAdapter:
        """Wrapper for CLIP prompt tuning + entropy-minimizing adaptation."""

        def __init__(
            self,
            class_names: Sequence[str],
            arch: str = "ViT-B-16",
            pretrained: str = "openai",
            precision: str = "fp32",
            ctx_init: Optional[str] = "a photo of a",
            n_ctx: int = 16,
            lr: float = 5e-4,
        ) -> None:
            if not class_names:
                raise ValueError("Provide at least one target label.")
            self.class_names = [name.strip() for name in class_names if name.strip()]
            if not self.class_names:
                raise ValueError("Labels cannot be empty strings.")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clip_model, _, preprocess = create_model_and_transforms(
                arch, pretrained=pretrained, device=self.device, precision=precision
            )
            self.model = clip_model.to(self.device)
            self.model.eval()
            self.preprocess = preprocess

            ctx_init_value = ctx_init.strip() if ctx_init and ctx_init.strip() else None
            self.prompt_learner = PromptLearner(
                self.model,
                arch,
                self.class_names,
                n_ctx=n_ctx,
                ctx_init=ctx_init_value,
                class_token_pos="end",
            ).to(self.device)
            self.text_encoder = TextEncoder(self.model).to(self.device)

            self.optimizer = torch.optim.Adam([self.prompt_learner.ctx], lr=lr)
            self.loss_fn = Entropy()

        def reset_prompt(self) -> None:
            with torch.no_grad():
                self.prompt_learner.reset()
            self.prompt_learner.to(self.device)
            self.optimizer = torch.optim.Adam(
                [self.prompt_learner.ctx],
                lr=self.optimizer.param_groups[0]["lr"],
            )

        def preprocess_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
            tensors = []
            for image in images:
                tensor = self.preprocess(image.convert("RGB"))
                tensors.append(tensor)
            batch = torch.stack(tensors, dim=0)
            return batch.to(self.device)

        def get_text_features(self) -> torch.Tensor:
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts.to(self.device)
            with torch.no_grad():
                text_feats = self.text_encoder(prompts, tokenized_prompts)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            return text_feats

        def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
            text_feats = self.get_text_features()
            image_feats = self.model.encode_image(image_tensor)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            return logit_scale * image_feats @ text_feats.T

        def adapt(self, image_tensor: torch.Tensor, steps: int) -> None:
            if steps <= 0:
                return

            self.prompt_learner.train()
            for _ in range(steps):
                logits = self.forward(image_tensor)
                loss = self.loss_fn(logits).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.prompt_learner.eval()

        def classify(
            self,
            images: Sequence[Image.Image],
            steps: int = 0,
            topk: int = 3,
        ) -> List[dict]:
            if not images:
                return []
            image_tensor = self.preprocess_images(images)
            if steps > 0:
                self.adapt(image_tensor, steps)

            with torch.no_grad():
                logits = self.forward(image_tensor)
            probs = logits.softmax(dim=1)

            top_k = min(topk, len(self.class_names))
            top_probs, top_idxs = probs.topk(top_k, dim=1)
            top_probs = top_probs.cpu()
            top_idxs = top_idxs.cpu()

            results = []
            for idx, image in enumerate(images):
                prediction = self.class_names[top_idxs[idx, 0]]
                confidence = float(top_probs[idx, 0])
                topk_list = [
                    (self.class_names[top_idxs[idx, j]], float(top_probs[idx, j]))
                    for j in range(top_k)
                ]
                results.append(
                    {
                        "image": image,
                        "prediction": prediction,
                        "confidence": confidence,
                        "top_k": topk_list,
                    }
                )
            return results


def _get_session_adapter(
    labels: Sequence[str],
    arch: str,
    precision: str,
    ctx_init: Optional[str],
    n_ctx: int,
    lr: float,
) -> BATCLIPAdapter:
    key = (
        tuple(labels),
        arch,
        precision,
        ctx_init or "",
        n_ctx,
        lr,
    )
    stored = st.session_state.get("batclip_adapter")
    if stored and stored["key"] == key:
        adapter: BATCLIPAdapter = stored["adapter"]
    else:
        adapter = BATCLIPAdapter(
            class_names=list(labels),
            arch=arch,
            pretrained="openai",
            precision=precision,
            ctx_init=ctx_init,
            n_ctx=n_ctx,
            lr=lr,
        )
        st.session_state["batclip_adapter"] = {"key": key, "adapter": adapter}
    adapter.reset_prompt()
    return adapter


def _safe_load_images(files: Sequence[st.uploaded_file_manager.UploadedFile]) -> List[Tuple[str, Image.Image]]:
    loaded: List[Tuple[str, Image.Image]] = []
    for file in files:
        try:
            image = Image.open(file)
            loaded.append((file.name, image.convert("RGB")))
        except Exception:
            continue
    return loaded


def main() -> None:
    st.set_page_config(page_title="BATCLIP Topic Classifier", layout="wide")

    st.title("BATCLIP custom topic classifier")
    st.markdown(
        "Enter topic labels, upload images, and optionally apply BATCLIP-style test-time "
        "adaptation (TTA) via prompt tuning."
    )

    if IMPORT_ERROR is not None:
        st.error(
            "This demo requires `open-clip-torch`. Install it with `pip install open-clip-torch` "
            f"before running. ({IMPORT_ERROR})"
        )
        st.stop()

    st.sidebar.header("Chọn nhãn")
    default_labels = ["cat", "dog", "pizza", "people"]
    api_key = DEFAULT_GENAI_API_KEY
    if not api_key:
        st.sidebar.caption(
            "Muốn tự sinh caption (tuỳ chọn): set biến môi trường `GEMINI_API_KEY` trước khi chạy demo."
        )

    # lưu nhãn tùy chỉnh trong session
    if "custom_labels" not in st.session_state:
        st.session_state.custom_labels = []

    selected_labels = []

    # 1) Hiển thị danh sách tick chung (mặc định + nhãn tự thêm)
    all_labels = list(default_labels) + list(st.session_state.custom_labels)
    for lbl in all_labels:
        default_on = (lbl in ["cat", "dog"]) if lbl in default_labels else True
        chk_key = f"chk_{lbl}"
        # Nếu widget key đã có trong session_state thì không set `value=` để tránh warning màu vàng
        if chk_key in st.session_state:
            checked = st.sidebar.checkbox(lbl, key=chk_key)
        else:
            checked = st.sidebar.checkbox(lbl, value=default_on, key=chk_key)
        if checked:
            selected_labels.append(lbl)

    # 2) Form thêm nhãn mới (đặt sau danh sách tick)
    with st.sidebar.form("add_label_form", clear_on_submit=True):
        new_label = st.text_input("Thêm nhãn mới", key="new_label_input")
        submitted = st.form_submit_button("Thêm nhãn")

    if submitted:
        lbl_new = new_label.strip()
        if not lbl_new:
            st.sidebar.warning("Nhãn mới trống.")
        elif lbl_new in default_labels or lbl_new in st.session_state.custom_labels:
            st.sidebar.info("Nhãn đã tồn tại.")
        else:
            st.session_state.custom_labels.append(lbl_new)
            # tự tick nhãn mới
            st.session_state[f"chk_{lbl_new}"] = True
            st.rerun()

    # cố định tham số model
    arch = "ViT-B-16"
    precision = "fp32"
    tta_steps = 4
    lr = 5e-4
    ctx_prefix = "a photo of a"
    n_ctx = 16

    uploaded = st.file_uploader(
        "Upload ảnh",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Nhấn 'Chạy phân loại' để áp dụng TTA lên ảnh đã upload.",
    )
    st.caption("Mẹo: Bạn có thể chọn nhiều ảnh trong hộp thoại bằng Ctrl (hoặc Shift) hoặc kéo-thả nhiều file cùng lúc.")
    if uploaded:
        st.caption(f"Đã chọn: {len(uploaded)} ảnh")

    if not selected_labels:
        st.warning("Cần chọn/nhập ít nhất một nhãn để phân loại.")
        return

    if not uploaded:
        st.info("Upload ít nhất một ảnh để xem kết quả.")
        return

    if st.button("Chạy phân loại"):
        with st.spinner("Tải model và chạy TTA..."):
            adapter = _get_session_adapter(
                selected_labels,
                arch,
                precision,
                ctx_init=ctx_prefix if ctx_prefix.strip() else None,
                n_ctx=n_ctx,
                lr=lr,
            )
            # Auto-generate captions for selected labels (if available)
            if GENAI_AVAILABLE and api_key:
                try:
                    out_path = ROOT_DIR / "LLM_Caption" / "class_dict.json"
                    if HAVE_SHARED_CAPTION_GEN:
                        shared_generate_captions(  # type: ignore[misc]
                            selected_labels,
                            api_key,
                            out_path,
                            temperature=0.6,
                        )
                    else:
                        _generate_llm_captions_inline(selected_labels, api_key, out_path, temperature=0.6)
                except Exception as exc:  # pragma: no cover
                    st.sidebar.warning(f"Sinh caption lỗi (bỏ qua): {exc}")
            image_pairs = _safe_load_images(uploaded)
            if not image_pairs:
                st.error("Không thể đọc ảnh. Vui lòng thử lại với file khác.")
                return
            names, pil_images = zip(*image_pairs)
            results = adapter.classify(pil_images, steps=tta_steps, topk=3)

        st.subheader("Kết quả phân loại")
        # nhóm theo nhãn
        grouped = {}
        for name, result in zip(names, results):
            grouped.setdefault(result["prediction"], []).append((name, result["image"]))

        for label, items in grouped.items():
            st.markdown(f"**Nhãn: {label}**")
            cols = st.columns(4)
            for idx, (fname, img) in enumerate(items):
                with cols[idx % 4]:
                    st.markdown(f"{fname}")
                    st.image(img, width=220)

        st.sidebar.success(f"Hoàn thành (device: {adapter.device})")


if __name__ == "__main__":
    main()

