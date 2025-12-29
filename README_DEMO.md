# README_DEMO — Hướng dẫn chạy demo `demo_app.py`

Demo này là một app Streamlit cho phép:
- Chọn/nhập nhãn (topic labels)
- Upload ảnh
- Chạy phân loại bằng CLIP + prompt tuning (BATCLIP-style TTA)
- (Tuỳ chọn) Tự sinh caption cho từng nhãn bằng Gemini và lưu vào `LLM_Caption/class_dict.json`

---

## Yêu cầu

- Python 3.9+ (khuyến nghị 3.10/3.11)
- Windows/macOS/Linux đều chạy được
- GPU là tuỳ chọn (CPU vẫn chạy, chỉ chậm hơn)

---

## Cài đặt nhanh (pip)

Từ thư mục repo, vào thư mục `batclip-coco`:

```bash
cd batclip-coco
```

Cài dependencies tối thiểu cho demo:

```bash
pip install -U pip
pip install streamlit open-clip-torch pillow
```

Nếu bạn muốn **tự sinh caption bằng Gemini** (tuỳ chọn):

```bash
pip install google-genai
```

> Ghi chú: `torch` thường sẽ được kéo theo khi cài `open-clip-torch`. Nếu máy bạn chưa có `torch` phù hợp (CPU/GPU), hãy cài theo hướng dẫn chính thức của PyTorch.

---

## Chạy demo

Trong thư mục `batclip-coco`:

```bash
streamlit run demo_app.py
```

Sau đó mở URL mà Streamlit in ra trong terminal (thường là `http://localhost:8501`).

---

## Tuỳ chọn: bật tự sinh caption bằng Gemini

Demo **không có ô nhập API key** trong UI. Nếu muốn tự sinh caption, bạn cần set biến môi trường `GEMINI_API_KEY` trước khi chạy.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
cd E:\CS406\demo\batclip-coco
streamlit run demo_app.py
```

### macOS/Linux (bash/zsh)

```bash
export GEMINI_API_KEY="YOUR_KEY"
cd batclip-coco
streamlit run demo_app.py
```

Khi có API key và bạn bấm **“Chạy phân loại”**, demo sẽ (tuỳ điều kiện) tự sinh caption và ghi/merge vào:
- `batclip-coco/LLM_Caption/class_dict.json`

---

## Gợi ý sử dụng

- **Chọn nhãn** ở sidebar (tick nhãn có sẵn hoặc thêm nhãn mới).
- **Upload 1 hoặc nhiều ảnh** (jpg/png).
- Bấm **“Chạy phân loại”** để xem kết quả nhóm theo nhãn dự đoán.

---

## Lỗi thường gặp

- **Không chạy được vì thiếu `open-clip-torch`**
  - Cài lại: `pip install open-clip-torch`

- **Muốn dùng Gemini nhưng báo thiếu `google-genai`**
  - Cài: `pip install google-genai`

- **Đã set `GEMINI_API_KEY` nhưng vẫn không sinh caption**
  - Kiểm tra biến môi trường đã set trong đúng terminal đang chạy Streamlit.
  - Demo chỉ chạy phần Gemini khi có `google-genai` + có `GEMINI_API_KEY` hợp lệ.

- **Chạy CPU rất chậm**
  - Bình thường với CLIP + prompt tuning. Nếu có GPU/CUDA, cài PyTorch bản CUDA để tăng tốc.


