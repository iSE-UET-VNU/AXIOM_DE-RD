# Phân tích từ first principles — cần gì, yếu ở đâu, làm gì tiếp

Viết sau khi 10 giả thuyết bị bác bỏ và 2 giả thuyết trụ lại. Mục đích là dừng
việc thử tiếp những thứ dữ liệu đã loại, và chỉ ra chính xác chỗ còn dư địa.

---

## 1. Hệ thống thực chất đang ước lượng cái gì

Bỏ hết thuật ngữ, retrieval ở đây là ước lượng `P(trang p liên quan | truy vấn q)`.
Ta đang ước lượng nó bằng bốn thứ:

| thành phần | bản chất | có được huấn luyện cho việc xếp hạng không? |
|---|---|---|
| BM25 | trùng lặp từ vựng | **không** — công thức thống kê thuần |
| dense (te3-small) | tương đồng vector đã pooling | **không** — embedder đa dụng, zero-shot |
| fusion α=0.7 | tổ hợp tuyến tính cố định | **không** — một hằng số |
| SEP | tiên nghiệm cấu trúc | **không** — số học trên điểm có sẵn |

**Đây là phát hiện nền tảng: không một thành phần nào trong pipeline được huấn
luyện để xếp hạng cho tác vụ này.** Tất cả đều zero-shot.

Điều đó giải thích toàn bộ kết quả của phiên làm việc:

| loại can thiệp | kết quả |
|---|---|
| sắp xếp lại tín hiệu zero-shot (SEP, DCW, PRF, MaxSim, union, multi-hop, routing) | **+0 đến +2.4** |
| thành phần **được huấn luyện** duy nhất (cross-encoder rerank) | **+5.08** |

Mọi thứ tôi thử đều là hoán vị số học của cùng một nhúm tín hiệu zero-shot. Trần
lý thuyết thì rất cao — oracle sắp xếp lại top-20 là **71.68** so với 43.86 hiện
tại — nhưng để chạm tới nó cần một **hàm liên quan được học**, không phải thêm
một công thức cộng trọng số nữa.

Bằng chứng phụ củng cố điều này: khi tôi thử dùng LLM sinh (gpt-4o-mini, gpt-4o)
làm bộ phán quyết liên quan cho DAT, chúng chỉ đạt **50.2% / 52.2% độ chính xác**
trên nền tỉ lệ cơ sở 35.3% — tức ngang mức ngẫu nhiên. Không phải "LLM là đủ",
mà phải là **mô hình được huấn luyện riêng cho xếp hạng**.

---

## 2. Ba điểm yếu, đo được

### 2.1. Không có mô hình xếp hạng được huấn luyện — điểm yếu lớn nhất

Trần: oracle reorder ở depth 20 = **71.68**, depth 50 = 83.88. Ta đang ở 43.86.
Cross-encoder duy nhất đã thử (Voyage rerank-2.5, depth 20) đưa lên **49.23**,
tức chỉ lấy được ~5 trong 27.5 điểm khả dụng — và bị chặn ở depth 20 vì hạn mức
free tier, không phải vì kỹ thuật.

### 2.2. Trọng số fusion cố định cho mọi truy vấn

α = 0.7 áp cho tất cả. Oracle chọn α theo từng truy vấn = **52.16** (+8.60).
Đã kiểm chứng đây là tín hiệu thật, không phải nhiễu chọn lọc: gán cho mỗi truy
vấn α tốt nhất **của một truy vấn khác** chỉ được 39.87 — thấp hơn mọi α cố định.
**113/302 truy vấn (37%) hợp với BM25 thuần**, ta đang ép chúng dùng trọng số
nghiêng về dense.

Đã thử 7 đặc trưng bề mặt của truy vấn để dự đoán α — tất cả r < 0.13. Thông tin
không nằm ở câu hỏi, nó nằm ở **tương tác giữa câu hỏi và corpus** — lại là thứ
mà một mô hình được huấn luyện mới nắm được.

### 2.3. Nội dung hình ảnh — chỗ ta trượt một cách hệ thống

Đây là phần mới đo, và nó thay đổi kết luận trước đó của tôi. Chia các trang gold
theo thứ hạng ta gán cho chúng:

| nhóm | n | số ký tự (trung vị) | **tỉ lệ nội dung từ hình** |
|---|---|---|---|
| tìm thấy (rank < 10) | 706 | 819 | **0.101** |
| trượt, còn trong top-100 | 999 | 815 | **0.170** |
| trượt hẳn (rank ≥ 100) | 473 | 818 | **0.246** |
| *toàn corpus (tham chiếu)* | 1674 | 765 | *0.191* |

**Gradient đơn điệu và sạch.** Quan trọng: số ký tự gần như bằng nhau
(819 / 815 / 818), nên **không** phải "trang trượt vì ít chữ". Trang ta trượt hẳn
có **tỉ lệ nội dung đến từ hình cao gấp 2.4 lần** trang ta tìm thấy.

Nói cách khác: **ta trượt đúng những trang mà thông tin nằm trong hình.**

Điều này hoà giải một kết quả tưởng như mâu thuẫn. Bỏ toàn bộ text của Figure chỉ
làm giảm 1.20 NDCG (không có ý nghĩa thống kê) — vì bản diễn giải hình thành chữ
của KDL yếu **ở mọi nơi**. Nhưng những trang *phụ thuộc* vào hình thì bị trượt một
cách hệ thống. Tín hiệu yếu đều khắp, nhưng thiệt hại dồn vào một nhóm cụ thể:
**473 trang gold bị trượt hẳn, chiếm 21.7% tổng số trang gold.**

---

## 3. Vậy visual có đáng thử không? — Có, và đây là lý do định lượng

### Cái đã thử và vì sao chưa kết luận được

Arm visual đầu tiên đã chạy: CLIP ViT-B/32 @224px trên 1,674 ảnh render tại chỗ.
Kết quả **4.45 NDCG@10** (text 43.86, ngẫu nhiên 0.62). Hợp nhất không giúp.

Nhưng phân rã theo loại truy vấn cho thấy nó hành xử **đúng bản chất**:
Infographic 7.72 (cao nhất) → Text 4.20 (thấp nhất). Nó nắm bố cục, **không đọc
được chữ** — đúng như kỳ vọng cho 224×224 trên trang dày chữ, model huấn luyện
tiếng Anh, truy vấn tiếng Pháp, dữ liệu huấn luyện là ảnh tự nhiên chứ không
phải tài liệu.

**Nên đây là "chưa đo được", không phải "visual không giúp".**

### Vì sao lần sau có cơ sở để tin hơn

1. **Nhóm mục tiêu tồn tại và đã định lượng được**: 473 trang gold trượt hẳn, với
   tỉ lệ nội dung từ hình 0.246 — gấp 2.4 lần nhóm tìm được.
2. **Bổ trợ, không phải thay thế.** Trên chính subset physics, các visual
   retriever đạt 43.2–48.5, còn text của ta 43.86 — **ngang nhau**. Nên visual
   không phải viên đạn bạc; giá trị của nó là **sửa đúng nhóm mà text trượt**.
   (Con số 91.0 của ColEmbed là ViDoRe **V1**, một benchmark dễ hơn nhiều — đừng
   dùng nó để kỳ vọng cho V3.)
3. **Bản rẻ là khả thi.** Bảng 6 của ColEmbed: bi-encoder ảnh **một vector** +
   reranker đạt 0.9064 so với 0.9106 của late interaction đầy đủ, ở **3.8 GB
   thay vì 10,311 GB / triệu trang**. Một vector/trang **không cần đổi index
   contract** — `LocalIndex.vectors` vẫn là mảng 2-D như hiện tại.

### Kỳ vọng thực tế

Nếu một VLM tài liệu lấy lại được ngay cả một phần ba trong 473 trang trượt hẳn,
đó là mức tăng lớn hơn tất cả những gì tôi làm được bằng cách sắp xếp lại tín
hiệu text. Nhưng nói cho ngay: visual đứng một mình trên physics chỉ ngang text,
nên **hãy kỳ vọng ở phần hợp nhất, không phải ở arm visual đơn lẻ**.

---

## 4. Cần gì — theo thứ tự đòn bẩy

| # | việc cần làm | đòn bẩy | rào cản còn lại |
|---|---|---|---|
| 1 | **Đổi generator** | **+27.73 điểm QA** (đã đo, p=0.0001) | hạn mức API |
| 2 | **Mô hình xếp hạng được huấn luyện** (cross-encoder tài liệu, chạy trên Colab) | +5.08 đã đo với Voyage; trần depth-20 là 71.68 | GPU Colab |
| 3 | **VLM tài liệu cho arm visual** (ColPali/ColQwen) | nhắm vào 473 trang trượt hẳn | GPU Colab — ảnh **đã render sẵn** |
| 4 | α thích ứng theo truy vấn | trần +8.60, chưa có bộ dự đoán | cần mô hình học tương tác |
| 5 | SEP | +2.41 (đã có, miễn phí) | không — đã chạy được |

**Điểm mấu chốt:** hạng 1 là một dòng cấu hình. Hạng 2 và 3 đều **chỉ cần Colab**,
và tôi đã gỡ phần lớn hạ tầng cho hạng 3 — ảnh render tại chỗ từ PDF gốc (không
cần tải 442 MB–2.2 GB), pipeline `render → embed → eval` đã chạy được đầu-cuối,
đổi model chỉ là đổi tham số `--repo`.

---

## 5. Điều nên thôi thử

Dữ liệu đã loại các hướng sau, kèm số liệu:

| hướng | kết quả |
|---|---|
| DCW — trừ centroid tài liệu | −5.73 / −9.50 |
| MaxSim dưới cấp trang | −1.18 đến −11.16 |
| tăng trọng số theo loại block | không có cơ sở (tỉ lệ 0.80–1.03) |
| PRF / Rocchio | −0.1 đến −1.8 |
| định tuyến theo văn phong truy vấn | r = 0.013 |
| hợp nhất bằng union hai nhánh | −5.72 |
| lan truyền nhiều bước | −2.00 / −6.64 |
| nới cửa sổ lân cận | phẳng hoặc âm |
| DAT với LLM sinh làm judge | judge ở mức ngẫu nhiên (50.2% / 52.2%) |
| analyzer tiếng Pháp | thật sự có ích (+1.65 tại α=0.5) nhưng bị SEP bao trùm |

Nguyên tắc chung rút ra: **tổng hợp thì thắng, định vị cục bộ thì thua** — trừ khi
việc định vị đó do một mô hình *được huấn luyện* thực hiện. Corpus này có bằng
chứng khuếch tán (7.21 trang gold/truy vấn, 57% số trang là gold của một truy vấn
nào đó), nên các phương pháp thiết kế cho bài toán "mò kim đáy bể" đều thua ở đây.

Chạy lại chẩn đoán mục 2.3:
```bash
python research/experiments/vidore_why_missed.py
```
