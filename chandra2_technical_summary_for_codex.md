# Chandra OCR 2 — Technical Summary and Implementation Audit Guide

> Mục đích: dùng tài liệu này làm đầu vào cho Codex để đối chiếu implementation Chandra2 hiện tại trong dự án ingestion/indexing.
>
> Thời điểm kiểm tra nguồn chính thức: **2026-07-16**. Nội dung được đối chiếu với repository `datalab-to/chandra` và model repository `datalab-to/chandra-ocr-2`. Implementation có thể thay đổi theo version; Codex cần kiểm tra version thực tế đang được pin trong dự án.

---

## 1. Kết luận ngắn

Chandra OCR 2 là một **generative Vision-Language Model (VLM)** dành cho document OCR. Model nhận một ảnh trang tài liệu cùng prompt, sau đó **sinh chuỗi token**, mặc định hướng tới structured HTML có thông tin layout. Nó không phải pipeline OCR cổ điển dạng detector → text recognizer.

Hệ thống open-source của Chandra gồm các phần độc lập:

1. **Model weights**: nằm trong model repository trên Hugging Face, mặc định là `datalab-to/chandra-ocr-2`.
2. **Package `chandra-ocr`**: chứa code orchestration, render PDF, xử lý ảnh, prompt template, adapter inference, parse output và CLI.
3. **Hugging Face backend**: load weights trực tiếp vào Python process rồi chạy inference trên GPU.
4. **vLLM backend**: giữ model trong một inference server riêng; các pipeline/client gửi HTTP request tới OpenAI-compatible endpoint.
5. **Pipeline của dự án**: nên chịu trách nhiệm validation, normalization về data contract, provenance, quality checks, retry cấp document và lưu raw artifacts.

Điểm cần nhớ:

- `chandra-ocr` **không đóng gói weights** trong Python package.
- Ở cả Hugging Face và vLLM, weights cuối cùng đều phải được tải về máy có GPU và load vào VRAM.
- `from openai import OpenAI` trong backend vLLM chỉ dùng OpenAI Python SDK làm **HTTP client**. Điều đó không có nghĩa OpenAI host Chandra2 hoặc OpenAI API key cho phép dùng Chandra2.
- Với implementation open-source hiện tại, model chủ yếu sinh **raw structured HTML**; package sau đó chuyển thành Markdown, cleaned HTML, layout chunks và metadata.

---

## 2. Phân biệt các thành phần

### 2.1 Model repository

Model ID mặc định:

```text
datalab-to/chandra-ocr-2
```

Model repository chứa các artifact như:

- model weights, thường ở dạng `.safetensors`;
- model configuration;
- tokenizer;
- multimodal/image processor;
- generation configuration;
- các file cần thiết để Transformers hoặc vLLM load model.

Model chính thức hiện được Datalab mô tả là Chandra OCR 2, một model OCR generative khoảng **4B tham số**, hỗ trợ tài liệu phức tạp, bảng, công thức, form, handwriting và hơn 90 ngôn ngữ.

### 2.2 Python package `chandra-ocr`

Package chứa code ứng dụng, không phải toàn bộ model weights. Các dependency chính cho thấy package đảm nhiệm:

- `pypdfium2`: render PDF;
- `Pillow`: xử lý ảnh;
- `beautifulsoup4`: parse HTML;
- `markdownify`: HTML → Markdown;
- `openai`: client gọi vLLM OpenAI-compatible API;
- `pydantic-settings`: configuration;
- optional `torch`, `transformers`, `accelerate`: direct Hugging Face inference;
- optional `streamlit`: web app.

CLI/package entry points hiện gồm:

```text
chandra
chandra_app
chandra_screenshot
chandra_vllm
```

### 2.3 Hugging Face Transformers

Hugging Face backend chịu trách nhiệm:

- tải hoặc đọc weights từ local cache;
- khởi tạo processor và model;
- đưa model lên GPU;
- tạo tensors từ ảnh và prompt;
- gọi `model.generate()`;
- decode output token thành raw text.

### 2.4 vLLM

vLLM chịu trách nhiệm:

- load model weights trên GPU server;
- giữ model sống trong một server process;
- cung cấp OpenAI-compatible HTTP API;
- scheduling nhiều request;
- batching và quản lý concurrency;
- KV cache và các tối ưu serving phù hợp với generative model;
- quản lý giới hạn GPU memory, sequence và batched tokens.

### 2.5 OpenAI Python SDK

Code kiểu sau:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://gpu-server:8000/v1",
    api_key="EMPTY",
)
```

có nghĩa là:

```text
Python client
    → HTTP request tới base_url do ta chỉ định
    → vLLM server tự host
    → Chandra2 trên GPU của ta
```

SDK chỉ serialize/deserialize request theo giao thức tương thích OpenAI. Request không đi tới OpenAI khi `base_url` trỏ về server riêng.

---

## 3. Hai kiến trúc inference chính

## 3.1 Direct Hugging Face inference

```text
Python pipeline process trên GPU server
    ├── load Chandra processor
    ├── load Chandra weights
    ├── render/đọc ảnh
    ├── model.generate()
    ├── post-process
    └── lưu output
```

Cài đặt:

```bash
pip install "chandra-ocr[hf]"
```

CLI:

```bash
chandra input.pdf ./output --method hf
```

Đặc điểm:

- model nằm trong cùng Python process với pipeline hoặc worker;
- phù hợp R&D, debugging, benchmark và batch đơn giản;
- không có network overhead;
- dễ can thiệp preprocessing/postprocessing;
- process nào load model thì process đó giữ một bản weights trong VRAM;
- hai terminal/process Python độc lập không tự chia sẻ cùng một object model hoặc CUDA memory.

Nếu terminal A chỉ load model bằng Hugging Face và terminal B chạy pipeline, terminal B không tự gọi được model của A. Muốn tách process, cần thêm một cơ chế giao tiếp như:

- FastAPI/HTTP;
- gRPC;
- Redis/RabbitMQ worker queue;
- multiprocessing queue/shared worker;
- Unix socket hoặc IPC khác.

Nếu không có service/IPC, cách đơn giản nhất là để pipeline và model chạy trong cùng process.

## 3.2 vLLM server inference

```text
Laptop / pipeline / ingestion worker
    → HTTP request
    → vLLM server trên GPU machine
    → Chandra2 inference
    → HTTP response
```

Khởi động qua wrapper chính thức:

```bash
pip install chandra-ocr
chandra_vllm
```

Hoặc tự chạy một vLLM server với model:

```bash
vllm serve datalab-to/chandra-ocr-2 \
  --host 0.0.0.0 \
  --port 8000
```

Đặc điểm:

- model nằm trong process vLLM riêng;
- laptop hoặc process khác có thể request tới GPU server;
- phù hợp khi nhiều worker hoặc thành viên dùng chung một GPU service;
- vLLM giải quyết sẵn API, concurrency và request scheduling;
- pipeline không cần cài PyTorch/model weights nếu chỉ đóng vai trò remote client;
- máy chạy vLLM vẫn phải có weights, CUDA và GPU phù hợp.

Trong môi trường lab, nên giới hạn endpoint trong private network hoặc dùng SSH tunnel:

```bash
ssh -L 8000:localhost:8000 user@gpu-server
```

Client trên laptop sau đó gọi:

```text
http://localhost:8000/v1
```

---

## 4. Weights được tải và lưu ở đâu?

### 4.1 Khi dùng Hugging Face direct

Lần đầu gọi:

```python
AutoModelForImageTextToText.from_pretrained(
    "datalab-to/chandra-ocr-2"
)
```

Transformers/Hugging Face Hub sẽ tải model artifacts về cache của user đang chạy process trên GPU server. Thư mục mặc định thường là:

```text
~/.cache/huggingface/hub/
```

Model có thể nằm dưới thư mục dạng:

```text
~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/
```

Lần chạy sau thường đọc lại cache, không tải lại toàn bộ từ Internet. Tuy nhiên, mỗi lần process khởi động, weights vẫn phải đi qua:

```text
Disk cache → RAM → VRAM
```

### 4.2 Khi dùng `chandra_vllm`

Wrapper hiện tại chạy Docker image vLLM và mount cache host:

```text
Host:      ~/.cache/huggingface
Container: /root/.cache/huggingface
```

Vì vậy model tải bởi vLLM container vẫn được giữ trên filesystem của GPU host và có thể tái sử dụng giữa các lần container khởi động.

Luồng:

```text
vLLM nhận MODEL_CHECKPOINT
    → Hugging Face Hub hoặc local model path
    → cache trên GPU host
    → load weights vào VRAM
    → serve endpoint
```

### 4.3 Tải trước và chạy offline/local path

Có thể tải model trước:

```bash
huggingface-cli download \
  datalab-to/chandra-ocr-2 \
  --local-dir /data/models/chandra-ocr-2
```

Sau đó dùng local path:

```bash
vllm serve /data/models/chandra-ocr-2 --port 8000
```

hoặc:

```python
AutoModelForImageTextToText.from_pretrained(
    "/data/models/chandra-ocr-2",
    local_files_only=True,
)
```

Có thể đổi cache root bằng `HF_HOME`, ví dụ:

```bash
export HF_HOME=/data/huggingface
```

---

## 5. Pipeline chi tiết của package `chandra-ocr`

```text
PDF / image / directory
    ↓
Input discovery
    ↓
PDF rendering hoặc image loading
    ↓
Image sizing / RGB conversion
    ↓
Prompt construction
    ↓
HF direct hoặc vLLM inference
    ↓
Raw structured HTML generated by model
    ↓
Layout parsing + cleaned HTML + Markdown
    ↓
Image/Figure cropping
    ↓
Merge pages
    ↓
Markdown / HTML / metadata JSON / image files
```

## 5.1 Input discovery

CLI hiện hỗ trợ:

```text
.pdf
.png
.jpg
.jpeg
.gif
.webp
.tiff
.bmp
```

Có thể xử lý:

- một file;
- toàn bộ file được hỗ trợ trong một thư mục;
- một page range cho PDF.

Source area:

```text
chandra/scripts/cli.py
get_supported_files()
```

## 5.2 Load ảnh

Source:

```text
chandra/input.py
load_image()
```

Hành vi chính:

1. mở file bằng Pillow;
2. convert sang RGB;
3. nếu width hoặc height nhỏ hơn `MIN_IMAGE_DIM`, resize bằng Lanczos;
4. giữ tỷ lệ ảnh.

Default hiện tại:

```text
MIN_IMAGE_DIM = 1536
```

## 5.3 Render PDF thành ảnh

Source:

```text
chandra/input.py
load_pdf_images()
flatten()
```

Hành vi chính:

1. mở PDF bằng `pypdfium2.PdfDocument`;
2. khởi tạo forms;
3. chọn trang theo page range;
4. flatten annotation/form field;
5. tính render scale dựa trên DPI và minimum page dimension;
6. render mỗi trang thành PIL image;
7. convert sang RGB.

Defaults hiện tại:

```text
IMAGE_DPI = 192
MIN_PDF_IMAGE_DIM = 1024
```

Package không thể hiện một image enhancement pipeline đầy đủ như:

- deskew;
- dewarp;
- denoise;
- deblur;
- adaptive binarization;
- contrast enhancement;
- orientation detection.

Nếu dự án cần các bước này, chúng nên nằm trong preprocessing layer riêng trước Chandra.

## 5.4 Scale ảnh cho model

Source:

```text
chandra/model/util.py
scale_to_fit()
```

Hàm này:

- kiểm tra ảnh hợp lệ;
- scale theo giới hạn pixel-area tối đa/tối thiểu;
- điều chỉnh width và height theo grid 28 pixel;
- cố gắng giữ aspect ratio;
- resize bằng Lanczos.

Defaults trong hàm hiện tại:

```text
max_size = (3072, 2048)
min_size = (1792, 28)
grid_size = 28
```

Lưu ý: implementation dùng tích của hai phần tử tuple để tính giới hạn tổng số pixel, không chỉ đơn giản clamp width/height riêng lẻ.

## 5.5 Prompt templates

Source:

```text
chandra/prompts.py
```

Hai prompt type chính:

```text
ocr
ocr_layout
```

### `ocr`

Yêu cầu OCR ảnh thành HTML.

### `ocr_layout`

Yêu cầu OCR ảnh thành HTML, chia thành các top-level layout block. Mỗi block là `div` với:

```html
<div
  data-label="Table"
  data-bbox="x0 y0 x1 y1"
>
  ...
</div>
```

Bounding box được chuẩn hóa trong không gian:

```text
0–1000
```

Các layout labels được prompt công bố gồm:

```text
Caption
Footnote
Equation-Block
List-Group
Page-Header
Page-Footer
Image
Section-Header
Table
Text
Complex-Block
Code-Block
Form
Table-Of-Contents
Figure
Chemical-Block
Diagram
Bibliography
Blank-Page
```

Prompt còn quy định:

- dùng tập HTML tag/attribute giới hạn;
- math ở dạng KaTeX-compatible LaTeX;
- table giữ `rowspan` và `colspan`;
- checkbox/radio được biểu diễn đúng;
- image có mô tả trong `alt`, không tự điền `src`;
- chart chuyển thành high-fidelity data;
- diagram chuyển thành Mermaid khi thích hợp;
- dòng text phải được nối thành paragraph với reading order tự nhiên;
- chemistry có thể dùng tag chuyên biệt.

`BatchInputItem` cũng cho phép truyền custom prompt thay vì prompt type.

## 5.6 Hugging Face backend

Source:

```text
chandra/model/hf.py
```

`load_model()` hiện làm các bước:

1. import Torch và Transformers;
2. load `AutoModelForImageTextToText` từ `MODEL_CHECKPOINT`;
3. dùng `torch.bfloat16`;
4. dùng `device_map="auto"` hoặc device được cấu hình;
5. có thể chọn attention implementation;
6. gọi `model.eval()`;
7. load `AutoProcessor`;
8. đặt tokenizer padding side là `left`;
9. attach processor vào model.

`generate_hf()` hiện:

1. chuyển mỗi `BatchInputItem` thành chat-style multimodal message;
2. scale ảnh;
3. thêm image trước, text prompt sau;
4. gọi processor `apply_chat_template()`;
5. chuyển tensors sang device của model;
6. cấu hình EOS và `<|im_end|>` làm stop tokens;
7. gọi `model.generate()`;
8. cắt input tokens khỏi sequence output;
9. batch decode;
10. trả về `GenerationResult(raw, token_count, error)`.

Default max output tokens:

```text
MAX_OUTPUT_TOKENS = 12384
```

## 5.7 vLLM client backend

Source:

```text
chandra/model/vllm.py
```

`generate_vllm()` hiện:

1. tạo `OpenAI` client với `VLLM_API_BASE` và `VLLM_API_KEY`;
2. dùng configured served model name hoặc gọi `/models` để lấy model đầu tiên;
3. scale ảnh;
4. encode ảnh thành PNG base64;
5. gửi ảnh dưới dạng data URL:

```text
data:image/png;base64,...
```

6. thêm text prompt;
7. gọi `client.chat.completions.create()`;
8. lấy `choices[0].message.content` làm raw output;
9. lấy completion token usage;
10. chạy nhiều request song song bằng `ThreadPoolExecutor`;
11. phát hiện output lặp token/substring;
12. retry lỗi hoặc repeated generation;
13. tăng temperature dần khi retry.

Defaults hiện tại:

```text
VLLM_API_BASE = http://localhost:8000/v1
VLLM_API_KEY = EMPTY
VLLM_MODEL_NAME = chandra
MAX_VLLM_RETRIES = 6
```

`EMPTY` chỉ là placeholder/default key của self-hosted server, không phải OpenAI API key.

## 5.8 Inference manager

Source:

```text
chandra/model/__init__.py
InferenceManager
```

Hành vi:

- `method="hf"`: load model ngay trong manager;
- `method="vllm"`: không load model local, chỉ dùng remote client;
- sau generation, luôn tạo `BatchOutputItem` bằng cách parse raw output.

Output cấp trang:

```python
BatchOutputItem(
    markdown=...,
    html=...,
    chunks=...,
    raw=...,
    page_box=...,
    token_count=...,
    images=...,
    error=...,
)
```

## 5.9 Parse raw output

Source:

```text
chandra/output.py
```

### Raw output

Raw output là chuỗi model sinh ra, kỳ vọng là structured HTML.

### `parse_html()`

Hàm này:

- parse bằng BeautifulSoup;
- chỉ xét top-level `div`;
- bỏ `Blank-Page`;
- mặc định bỏ `Page-Header` và `Page-Footer`;
- có thể bỏ `Image` và `Figure`;
- thêm `src` cho image/figure block;
- xóa `<img>` không có `src` trong non-image block;
- bọc plain text của `Text` block bằng paragraph nếu cần;
- trả về cleaned inner HTML.

Quan trọng: cleaned HTML được ghép từ **content bên trong** top-level div, nên layout wrapper và `data-label`/`data-bbox` không còn trong HTML cuối. Layout metadata nằm ở `chunks`.

### `parse_markdown()`

Luồng:

```text
raw HTML
    → parse_html()
    → custom Markdownify
    → Markdown
```

Xử lý đặc biệt gồm:

- ATX headings;
- list bullets;
- escaping ký tự Markdown;
- inline math `$...$`;
- block math `$$...$$`;
- table được giữ dưới dạng HTML table thay vì ép thành pipe table;
- links, superscript, subscript và pre/code.

### `parse_layout()` và `parse_chunks()`

Mỗi top-level layout div được chuyển thành:

```json
{
  "bbox": [x0, y0, x1, y1],
  "label": "Table",
  "content": "<table>...</table>"
}
```

Quá trình:

1. lấy `data-label`;
2. lấy normalized `data-bbox`;
3. scale bbox từ 0–1000 về pixel coordinates của ảnh gốc;
4. clamp vào image bounds;
5. lấy inner HTML;
6. loại nested `data-bbox` khỏi content;
7. serialize dataclass thành dictionary.

### `extract_images()`

Hàm tìm các chunk có label:

```text
Image
Figure
```

sau đó:

- tìm `<img>` trong chunk content;
- dùng bbox đã scale để crop page image;
- tạo tên `.webp` từ hash của HTML và index;
- trả về dictionary tên file → PIL image.

## 5.10 Merge và lưu output

Source:

```text
chandra/scripts/cli.py
save_merged_output()
```

CLI:

- xử lý page theo batch;
- dùng `prompt_type="ocr_layout"`;
- merge Markdown của các trang;
- merge HTML của các trang;
- có optional page separator;
- cộng token count, chunk count và image count;
- lưu cropped images;
- lưu metadata theo trang.

Output structure mặc định:

```text
output/
└── document/
    ├── document.md
    ├── document.html
    ├── document_metadata.json
    ├── <hash>_<index>_img.webp
    └── ...
```

Metadata JSON mặc định gần dạng:

```json
{
  "file_name": "document.pdf",
  "num_pages": 3,
  "total_token_count": 12000,
  "total_chunks": 45,
  "total_images": 4,
  "pages": [
    {
      "page_num": 0,
      "page_box": [0, 0, 2048, 3072],
      "token_count": 4000,
      "num_chunks": 15,
      "num_images": 1
    }
  ]
}
```

### Quan trọng về JSON

Mặc dù Chandra được mô tả là xuất Markdown, HTML và JSON, CLI open-source hiện tại mặc định chỉ ghi file `*_metadata.json` chứa summary metadata. `BatchOutputItem.chunks` có structured layout data nhưng CLI không serialize toàn bộ chunks thành một document JSON hoàn chỉnh.

Nếu dự án cần JSON contract đầy đủ, nên tự serialize:

```python
{
    "document_id": ...,
    "pages": [
        {
            "page_number": ...,
            "width": ...,
            "height": ...,
            "blocks": result.chunks,
            "raw": result.raw,
            "markdown": result.markdown,
            "html": result.html,
            "token_count": result.token_count,
            "error": result.error,
        }
    ]
}
```

---

## 6. `chandra_vllm` wrapper hiện làm gì?

Source:

```text
chandra/scripts/vllm.py
```

Wrapper hiện không chỉ gọi một Python function vLLM đơn giản; nó build và chạy một Docker command với image:

```text
vllm/vllm-openai:v0.17.0
```

Các behavior đáng chú ý:

- chọn GPU devices từ `VLLM_GPUS`;
- mount Hugging Face cache host vào container;
- publish port `8000:8000`;
- dùng IPC host;
- load `MODEL_CHECKPOINT`;
- dùng `bfloat16`;
- đặt max model length 18000;
- đặt GPU memory utilization 0.85;
- bật prefix caching;
- cấu hình multimodal min/max pixels;
- đặt served model name là `VLLM_MODEL_NAME`;
- scale `max-num-batched-tokens` và `max-num-seqs` theo VRAM profile;
- baseline tuning là H100 80GB;
- có preset cho H100, A100, L40S, A10, L4, RTX 4090, RTX 3090 và T4.

Do wrapper gọi `sudo docker run`, deployment hiện tại cần kiểm tra:

- user có quyền `sudo` hay không;
- Docker và NVIDIA Container Runtime đã được cài;
- port 8000 có bị chiếm không;
- firewall/private network;
- cache mount có đúng home user không;
- container version có tương thích driver/CUDA không.

Repository có issue báo rằng prefix cache có thể gần như không hit với Chandra2 do request đặt image trước prompt. Vì vậy, không nên mặc định coi `--enable-prefix-caching` chắc chắn mang lại lợi ích; cần benchmark trên workload thật.

---

## 7. So sánh trách nhiệm

| Thành phần | Trách nhiệm chính |
|---|---|
| Chandra2 weights | Visual understanding, OCR, reading order, layout reasoning và sinh raw structured HTML |
| `chandra-ocr` | Input loading, PDF rendering, image sizing, prompt templates, backend adapters, parse HTML/Markdown/chunks, crop images, CLI và output files |
| Transformers/PyTorch | Load model/processor và direct generation trên GPU |
| vLLM | Model serving, HTTP API, scheduling, concurrency, batching và GPU runtime |
| OpenAI SDK | Client giao tiếp với OpenAI-compatible vLLM endpoint |
| Dự án ingestion | Contract chuẩn, validation, provenance, quality score, retry policy, observability, storage và adapter giữa các parser |

---

## 8. Đề xuất integration contract cho dự án

Không nên để downstream phụ thuộc trực tiếp vào `BatchOutputItem` hoặc output CLI. Nên có adapter riêng:

```text
Chandra BatchOutputItem
    → ChandraAdapter
    → ParsedDocument contract chuẩn
```

Ví dụ contract:

```json
{
  "document_id": "...",
  "source": {
    "uri": "...",
    "mime_type": "application/pdf",
    "checksum": "..."
  },
  "parser": {
    "name": "chandra",
    "package_version": "...",
    "model_id": "datalab-to/chandra-ocr-2",
    "model_revision": "...",
    "backend": "hf|vllm",
    "config": {}
  },
  "pages": [
    {
      "page_number": 1,
      "width": 2048,
      "height": 3072,
      "markdown": "...",
      "html": "...",
      "raw_model_output": "...",
      "blocks": [
        {
          "id": "...",
          "type": "table",
          "bbox": {
            "x0": 10,
            "y0": 20,
            "x1": 1000,
            "y1": 800,
            "coordinate_space": "pixel"
          },
          "content_html": "<table>...</table>",
          "content_text": "...",
          "reading_order": 0,
          "assets": []
        }
      ],
      "metrics": {
        "output_tokens": 4000,
        "latency_ms": null,
        "error": false
      }
    }
  ],
  "errors": [],
  "warnings": []
}
```

Nên lưu đồng thời:

- raw model output;
- normalized structured output;
- Markdown convenience view;
- HTML convenience view;
- source image/page dimensions;
- parser/model version và revision;
- prompt type hoặc hash prompt;
- inference configuration;
- latency/token usage;
- retry/error history.

---

## 9. Các điểm Codex cần kiểm tra kỹ trong implementation hiện tại

### A. Version và source of truth

- [ ] `chandra-ocr` version nào đang được pin?
- [ ] Project dùng PyPI release, Git commit hay source copy?
- [ ] Model ID có đúng `datalab-to/chandra-ocr-2` không?
- [ ] Có pin model revision/commit hash không?
- [ ] vLLM version và Transformers version có tương thích model không?
- [ ] Config thực tế có override `local.env` hoặc environment variables không?

### B. Kiến trúc process

- [ ] Pipeline direct HF có load model đúng một lần hay load lại theo file/page?
- [ ] Có vô tình chạy nhiều process, mỗi process giữ một bản model trong VRAM không?
- [ ] Nếu inference tách process, giao tiếp dùng HTTP, queue hay IPC nào?
- [ ] Nếu vLLM, endpoint có health check và readiness check không?
- [ ] Client có timeout, retry và cancellation hợp lý không?

### C. Weights và cache

- [ ] Weights tải vào cache của user/container nào?
- [ ] `HF_HOME` có được cấu hình vào ổ đủ dung lượng không?
- [ ] Docker có mount cache bền vững không?
- [ ] Có tải lại model sau mỗi container recreation không?
- [ ] Có theo dõi model revision để đảm bảo reproducibility không?

### D. PDF và page indexing

- [ ] Kiểm tra semantics của `--page-range`.
- [ ] Code hiện parse chuỗi như `1-5` thành các số nguyên rồi so sánh trực tiếp với loop page index bắt đầu từ 0. Cần xác minh có off-by-one so với CLI expectation hay không.
- [ ] Page number trong contract của dự án là zero-based hay one-based?
- [ ] Annotation/form flatten có phù hợp mọi PDF không?
- [ ] Có xử lý encrypted/corrupt PDF không?
- [ ] Có giới hạn số trang, kích thước và timeout không?

### E. Image preprocessing

- [ ] Có resize hai lần không: một lần ở `load_image`, một lần ở `scale_to_fit`?
- [ ] Có mất chi tiết chữ nhỏ sau resize không?
- [ ] Có cần deskew/dewarp/denoise trước model không?
- [ ] Có detect orientation không?
- [ ] Có lưu ảnh đã render/preprocess để trace lỗi không?

### F. Prompt và output stability

- [ ] Dùng `ocr` hay `ocr_layout`?
- [ ] Có sửa prompt template không?
- [ ] Có lưu prompt version/hash không?
- [ ] Có validate output chỉ chứa allowed tags/attributes không?
- [ ] Có chống prompt injection từ nội dung tài liệu không?
- [ ] Có xử lý output bị truncate do `MAX_OUTPUT_TOKENS` không?
- [ ] Có phát hiện malformed HTML hoặc missing top-level div không?

### G. Layout and bounding boxes

- [ ] Bbox scale có đúng 1000 không?
- [ ] Bbox đã được clamp và validate `x0 < x1`, `y0 < y1` chưa?
- [ ] Coordinate system có được ghi rõ là pixel hay normalized không?
- [ ] Khi bbox invalid, implementation hiện comment “defaulting to full image” nhưng giá trị fallback là `[0,0,1,1]` trước khi scale, tương ứng vùng rất nhỏ chứ không phải full image. Cần quyết định behavior đúng và sửa nếu cần.
- [ ] Nested bbox bị strip khỏi chunk content có làm mất thông tin cần downstream không?
- [ ] Reading order có dựa hoàn toàn vào thứ tự div do model sinh không?

### H. HTML và Markdown

- [ ] Cleaned HTML hiện bỏ top-level layout wrappers; downstream có đang nhầm cleaned HTML vẫn còn bbox/label không?
- [ ] Markdown giữ table dưới dạng raw HTML table; downstream Markdown parser có hỗ trợ không?
- [ ] Có sanitize HTML trước khi hiển thị trong UI không?
- [ ] Có giữ raw HTML model output riêng không?

### I. JSON output

- [ ] Project có hiểu nhầm `*_metadata.json` là full structured document JSON không?
- [ ] Có serialize `result.chunks` không?
- [ ] Có schema version cho JSON contract không?
- [ ] Có validate JSON bằng Pydantic/JSON Schema không?
- [ ] `BatchOutputItem.chunks` được type annotate là `dict` nhưng runtime là list các dict; code dự án có dựa sai vào type annotation này không?

### J. Image/Figure extraction

- [ ] `extract_images()` chỉ crop khi chunk label là `Image`/`Figure` và inner content có `<img>`.
- [ ] `parse_html()` có thể thêm `<img>` nếu thiếu, nhưng `extract_images()` đọc raw chunks trước cleaned HTML; vì vậy figure không có `<img>` từ model có thể không được crop. Cần test thực tế.
- [ ] `InferenceManager.generate()` vẫn gọi `extract_images()` ngay cả khi `include_images=False`; flag chủ yếu ảnh hưởng cleaned HTML/Markdown và việc CLI save images. Cần xác minh behavior mong muốn.
- [ ] Có collision hoặc deduplication asset names giữa nhiều page không?
- [ ] Có lưu quan hệ block → asset filename không?

### K. vLLM client/server

- [ ] API key hiện là auth token tự host hay chỉ `EMPTY`?
- [ ] Không dùng nhầm OpenAI API key và không gửi request nhầm về OpenAI endpoint.
- [ ] Server bind interface và firewall có an toàn không?
- [ ] `chandra_vllm` dùng `sudo docker`; môi trường deployment có cho phép không?
- [ ] GPU profile được chọn đúng (`--gpu`) chưa?
- [ ] T4/A10/L4 có đủ VRAM và throughput cho workload không?
- [ ] `max-num-seqs`, `max-num-batched-tokens`, max model length và GPU utilization đã benchmark chưa?
- [ ] Prefix caching có thực sự hit trên workload Chandra không?
- [ ] Client thread pool có tạo quá nhiều request đồng thời không?
- [ ] Retry tăng temperature có làm output kém deterministic không?
- [ ] Base64 PNG có tạo overhead memory/network lớn không?

### L. Reliability và observability

- [ ] Có latency per page và per document không?
- [ ] Có token count, retry count và error category không?
- [ ] Có phân biệt model error, HTTP error, malformed output và normalization error không?
- [ ] Có idempotency và resume theo page không?
- [ ] Có dead-letter queue cho page lỗi không?
- [ ] Có lưu sample failure để benchmark regression không?

### M. License

- [ ] Code package là Apache-2.0.
- [ ] Model weights dùng modified OpenRAIL-M với điều kiện sử dụng riêng.
- [ ] Cần kiểm tra trường hợp commercial/self-host deployment của dự án có phù hợp license không.

---

## 10. Lệnh kiểm tra nhanh trên môi trường dự án

### Kiểm tra package/version

```bash
pip show chandra-ocr
pip freeze | grep -Ei 'chandra|torch|transformers|vllm|pypdfium'
```

### Kiểm tra settings thực tế

```bash
python - <<'PY'
from chandra.settings import settings
print(settings.model_dump())
PY
```

### Kiểm tra source package đang import

```bash
python - <<'PY'
import chandra
import inspect
from chandra.model import InferenceManager
print(chandra.__file__)
print(inspect.getfile(InferenceManager))
PY
```

### Kiểm tra Hugging Face cache

```bash
find ~/.cache/huggingface/hub -maxdepth 1 -iname '*chandra*' -print
du -sh ~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2 2>/dev/null
```

### Kiểm tra GPU/process

```bash
nvidia-smi
nvidia-smi pmon -c 1
```

### Kiểm tra vLLM endpoint

```bash
curl http://localhost:8000/v1/models
```

### Kiểm tra container

```bash
docker ps
docker logs <container-id> --tail 200
```

### Tìm integration hiện tại trong codebase

```bash
grep -RInE \
  'chandra|InferenceManager|BatchInputItem|generate_hf|generate_vllm|VLLM_API_BASE|MODEL_CHECKPOINT' \
  .
```

---

## 11. Prompt đề xuất để đưa cho Codex

```text
Read `chandra2_technical_summary_for_codex.md` and audit the current project implementation against it.

Requirements:
1. Inspect the actual installed/pinned `chandra-ocr`, Transformers, Torch and vLLM versions. Do not assume the repository master behavior matches the installed version.
2. Locate every Chandra integration point in the codebase: input loading, PDF rendering, preprocessing, prompt construction, HF/vLLM inference, retries, output parsing, normalization and persistence.
3. Produce a mapping table with columns:
   - Concern
   - Current project file/function
   - Current behavior
   - Expected/reference behavior
   - Match / Partial / Gap / Bug risk
   - Evidence
   - Recommended change
4. Pay special attention to:
   - repeated model loading and multiple GPU copies;
   - Hugging Face cache/model revision;
   - zero-based vs one-based PDF page range;
   - invalid bbox fallback;
   - loss of layout metadata in cleaned HTML;
   - whether full chunks are serialized to JSON;
   - Image/Figure cropping when the model omits an img tag;
   - include_images behavior;
   - vLLM endpoint/auth/network configuration;
   - retries that change temperature;
   - timeouts, observability and error recovery;
   - license implications.
5. Run or propose focused tests for each suspected mismatch. Do not modify code before presenting the audit unless explicitly asked.
6. End with a prioritized action plan:
   - P0 correctness/data-loss issues
   - P1 reliability/reproducibility issues
   - P2 performance/maintainability improvements
```

---

## 12. Official references

- Chandra source repository: https://github.com/datalab-to/chandra
- Chandra OCR 2 model card: https://huggingface.co/datalab-to/chandra-ocr-2
- Package configuration: https://github.com/datalab-to/chandra/blob/master/pyproject.toml
- Input/PDF rendering: https://github.com/datalab-to/chandra/blob/master/chandra/input.py
- Prompt templates: https://github.com/datalab-to/chandra/blob/master/chandra/prompts.py
- Output parsing: https://github.com/datalab-to/chandra/blob/master/chandra/output.py
- Hugging Face backend: https://github.com/datalab-to/chandra/blob/master/chandra/model/hf.py
- vLLM client backend: https://github.com/datalab-to/chandra/blob/master/chandra/model/vllm.py
- Inference manager: https://github.com/datalab-to/chandra/blob/master/chandra/model/__init__.py
- Data structures: https://github.com/datalab-to/chandra/blob/master/chandra/model/schema.py
- CLI/output writer: https://github.com/datalab-to/chandra/blob/master/chandra/scripts/cli.py
- vLLM Docker launcher: https://github.com/datalab-to/chandra/blob/master/chandra/scripts/vllm.py
- Settings: https://github.com/datalab-to/chandra/blob/master/chandra/settings.py
- Prefix caching issue: https://github.com/datalab-to/chandra/issues/89

---

## 13. Final mental model

```text
                Hugging Face Hub
                       │
                       │ model weights/config/processor
                       ▼
              GPU machine local cache
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  HF direct process           vLLM server process
  pipeline + model            model + HTTP API
          │                         ▲
          │                         │ HTTP
          ▼                         │
  raw structured HTML       pipeline/client process
          │                         │
          └────────────┬────────────┘
                       ▼
               chandra postprocess
          HTML / Markdown / chunks / images
                       │
                       ▼
              project normalization
                       │
                       ▼
              ParsedDocument contract
```

Chandra package cung cấp một document OCR pipeline cơ bản khá đầy đủ, nhưng chưa thay thế data contract và quality/reliability layer của hệ thống ingestion. Integration tốt nhất là cô lập Chandra phía sau một adapter, lưu raw output để trace, và chuẩn hóa tất cả output về contract ổn định của dự án.
