# Bàn giao phiên làm việc — Retrieval Vật Lý ViDoRe V3

Đọc file này trước khi làm gì tiếp. Mục đích: người/phiên tiếp theo biết ngay đang đứng ở đâu,
không cần đọc lại toàn bộ lịch sử. Ledger kỹ thuật đầy đủ (số liệu, phương pháp, mọi refutation)
là [`docs/vidore_v3_results.md`](vidore_v3_results.md) — file này chỉ là bản đồ + nhật ký thời gian.

Cập nhật lần cuối: **27/08/2026, 10:20**.

---

## 1. Trạng thái git

- Branch: `feature/retrieval-baseline`
- So với `origin/feature/retrieval-baseline`: **3 commit chưa push**, 0 behind (đã `git fetch`
  xác nhận lúc viết file này — `00e8bdc`, `d79ca66`, `13b651a`). Nhánh gốc đã đồng bộ tốt hơn dự
  kiến trước đó trong phiên (từng là 40 commit lệch).
- `.claude/` là untracked, không thuộc về nghiên cứu, không cần commit.
- Việc cần làm: `git push origin feature/retrieval-baseline` trước khi bắt đầu bất kỳ Colab
  notebook nào dùng `git clone` (notebook ColQwen2 hiện dùng `git archive` cục bộ nên **không**
  cần push, nhưng vẫn nên push để mọi thứ đồng bộ).

---

## 2. Kết quả đã xác nhận — sẵn sàng dùng ngay

| # | Cải tiến | NDCG@10 | Δ | p | Chi phí | Trạng thái |
|---|---|---:|---:|---:|---|---|
| 1 | SEP (rescoring theo file/trang) | 46.27 | +2.41 | 0.0004 | miễn phí, +0.3ms/câu | ✅ xong |
| 2 | Voyage rerank-2.5, depth-20 | 49.23 | +5.08 | 0.0001 | API, ~100 phút/302 câu, giới hạn free-tier | ✅ xong |
| 3 | ColQwen2 (visual, hợp nhất w≈0.7) | 47.37 | +3.22 | 0.0014 | GPU Colab (một lần), sau đó miễn phí | ✅ xong |
| 4 | Đổi generator → gpt-5.2 (QA, không phải NDCG) | Correct 71.67% | +27.73pp | 0.0001 | API — **đang bị chặn ngân sách** | ⏸ n=120/302 |

**Lưu ý quan trọng:** #1, #2, #3 đo trên baseline hơi khác nhau (43.86 / 44.15 / 44.15) và
**chưa từng được đo cùng lúc trên cùng một pool**. Đây là việc có giá trị nhất còn lại — không
cần hạ tầng mới, chỉ cần chạy SEP → rerank → ColQwen2 tuần tự trên cùng `physics_served_pool.json`.

---

## 3. Đã loại bỏ — đừng thử lại

| Hướng | Kết quả | Vì sao thất bại |
|---|---|---|
| DCW (trừ centroid tài liệu) | −5.7 / −9.5 | sai dấu ngay từ giả thuyết |
| MaxSim dưới cấp trang | −1.2 .. −11.2 | phân mảnh, không phải chất lượng nhóm |
| Trọng số theo loại block | vô căn cứ | tỉ lệ 0.80–1.03, không phân biệt được |
| PRF / Rocchio | −0.1 .. −1.8 | |
| Định tuyến theo văn phong câu hỏi | r = 0.01 | tín hiệu nằm ở tương tác câu hỏi–corpus, không phải bề mặt câu hỏi |
| Hợp nhất union hai nhánh | −5.7 | fusion đã khai thác complementarity tốt hơn union |
| Lan truyền nhiều bước (multi-hop) | −2.0 / −6.6 | khuếch tán nhiễu |
| DAT với LLM sinh làm giám khảo | ngẫu nhiên (~51% so với base rate 35.3%) | judge cần được huấn luyện, không phải LLM chung chung |
| CLIP visual (224px, tự nhiên-ảnh) | 4.45 vs text 43.86, fusion không giúp | model quá yếu, không đọc được chữ trang dày |
| **Rerank cục bộ (bge-reranker-base)** | **−2.9 (depth-20) đến −6.6 (depth-100)** | **model yếu — "có huấn luyện" ≠ "tốt"; xem §19 ledger** |

---

## 4. Nhật ký thời gian chạy — mọi thí nghiệm trong phiên này

⚠️ **Trước ngày 26/8, thời gian chạy KHÔNG được ghi lại có hệ thống** — chỉ có mtime file kết
quả (thời điểm hoàn tất), không có "bắt đầu → kết thúc" rõ ràng. Từ mục ColQwen2 / rerank cục bộ
trở đi, có log thời gian chính xác vì tôi quan sát trực tiếp qua `ps`/background task. Nên áp
dụng thói quen này cho mọi thí nghiệm dài (>1 phút) từ giờ: in timestamp bắt đầu/kết thúc trong
script, hoặc ghi chú thời lượng ngay khi báo cáo.

| Thí nghiệm | Bắt đầu | Kết thúc | Thời lượng | Ghi chú |
|---|---|---|---|---|
| Voyage rerank-2.5, depth-20, 302 câu | — | 14/08 00:46 | **~100 phút** | ghi trong docstring `physics_rerank_voyage.py`; giới hạn cứng bởi free-tier 3 RPM/10K TPM |
| CLIP visual arm (render 1674 ảnh + encode CPU/ONNX) | — | 25/08 21:20 | không log chính xác | chỉ có mtime kết quả |
| DAT LLM judge (2 model, graded+binary) | 25/08 ~19:33 | 25/08 ~19:38 | **~5 phút** | nhanh vì chỉ chấm top-1 mỗi leg, không phải cả pool |
| SEP end-to-end QA (baseline + SEP, 302 câu) | 25/08 19:48 | 25/08 19:54 | **~6 phút** | generation + judge cho 2 arm |
| Generator sweep (3 model, n=120) | 25/08 20:03 | 25/08 20:07 | **~4 phút** | n nhỏ, có thể tận dụng cache một phần |
| Full 302-query gpt-5.2 E2E | 25/08 tối | — | **THẤT BẠI giữa chừng** | key OpenRouter hết hạn mức tháng, 244/302 lỗi HTTP 403 |
| **Rerank cục bộ depth-100 (đoạn 1)** | 26/08 20:31 | 26/08 ~23:02 (bị dừng) | **2h31m**, đạt 240/302 | tiến trình bị kill khi CLI session kết thúc — **không có completion record**, phải resume từ checkpoint |
| **Rerank cục bộ depth-100 (đoạn 2, resume)** | 27/08 09:15 | 27/08 10:15 | **~1h**, hoàn tất 302/302 | resume từ checkpoint 240/302, không chạy lại từ đầu |
| **Tổng thời gian CPU thực tế** (không tính khoảng trống qua đêm) | | | **~3h31m** | cho 1 lần chạy đầy đủ 302 câu × depth-100 |
| `physics_rerank_local_depth_sweep.py` | 27/08 10:17 | 27/08 10:17 | **<1 phút** | đọc lại cache đã có, không tính toán lại |
| ColQwen2 — render + encode 1674 ảnh + 302 câu + score matrix | (Colab, không log phía tôi) | 26/08 15:13 (mtime export) | **không đo được chính xác** | chạy trên Colab của bạn; khoảng cách 15:13 → 22:15 (lúc bạn gửi lại) là thời gian thao tác của bạn (bao gồm cả lần sửa lỗi torchao), không phải thời gian GPU |
| `physics_colqwen_eval.py` (sau khi sửa bug O(n²)) | 26/08 22:24 | 26/08 22:24 | **<1 phút** | trước khi sửa bug, bị treo >120s và phải kill |

**Bài học rút ra, nên áp dụng tiếp:** hai lần chạy dài nhất phiên này (rerank Voyage 100 phút,
rerank cục bộ 3h31m) đều là **gọi API bị giới hạn tốc độ** hoặc **suy luận CPU không có GPU** —
cả hai đều là hệ quả trực tiếp của ràng buộc "chỉ có Colab cho GPU, key OpenRouter có hạn mức".
Không có thí nghiệm nào trong phiên này chạy lâu vì thuật toán chậm.

---

## 5. Việc cần làm tiếp — theo thứ tự ưu tiên

1. **Push 3 commit lên origin** — việc đầu tiên, 1 dòng lệnh.
2. **Hợp nhất SEP × ColQwen2 × Voyage rerank trên cùng một pool** — giá trị cao nhất, không cần
   hạ tầng mới, chỉ cần một script mới đọc cả 3 nguồn điểm số đã cache.
3. **Generator swap gpt-5.2, đủ 302 câu** — +27.73pp đã đo ở n=120, bị chặn bởi ngân sách
   OpenRouter (key giáo viên, đã hết hạn mức tháng). Chờ ngân sách mới.
4. **Tìm reranker cục bộ mạnh hơn** (nếu muốn tiếp tục hướng "rerank miễn phí, không API") —
   `bge-reranker-base` đã bị loại; thử `BAAI/bge-reranker-v2-m3` (lớn hơn, mới hơn) nếu có bản
   ONNX, nhưng **đo lại từ đầu**, đừng giả định "trained thì tốt hơn untrained" — §19 ledger là
   phản ví dụ trực tiếp.
5. **α thích ứng theo truy vấn** — trần +8.60 đã xác nhận là tín hiệu thật, nhưng chưa có bộ dự
   đoán rẻ nào hoạt động (7 đặc trưng, r < 0.13). Cần một mô hình học tương tác câu hỏi–corpus,
   chưa có hướng cụ thể.

---

## 6. Vị trí file quan trọng

| Loại | File |
|---|---|
| Ledger kỹ thuật đầy đủ (19 mục, số liệu + phương pháp) | `docs/vidore_v3_results.md` |
| Giải thích SEP tiếng Việt, có ví dụ thật từ KDL | `docs/sep_giai_thich.md` |
| Phân tích first-principles tiếng Việt | `docs/phan_tich_first_principles.md` |
| Notebook Colab ColQwen2 (đã sửa 2 lỗi: private-repo clone → git archive, torchao) | `ColQwen2_visual_arm_physics.ipynb` |
| Script rerank cục bộ + depth sweep | `research/experiments/physics_rerank_local.py`, `physics_rerank_local_depth_sweep.py` |
| Script eval ColQwen2 (đọc export từ Colab, không cần GPU) | `research/experiments/physics_colqwen_eval.py` |
| Artifact báo cáo cho trưởng nhóm (đã cập nhật đến §19) | https://claude.ai/code/artifact/bd820616-d28c-4813-ae82-0a7351204a01 |

---

## 7. Ràng buộc đang đứng — không tự ý vượt qua

- **Hết ngân sách OpenRouter** (key giáo viên, hết hạn mức tháng) — không tự ý chạy thêm gì cần
  API mà chưa hỏi lại.
- GPU chỉ có qua Google Colab (không có CUDA/MPS cục bộ) — mọi arm cần GPU phải đóng gói thành
  notebook, theo đúng pattern `git archive` + upload zip (không dùng `git clone` vì repo private).
- torch cục bộ pin ở 2.2.2 (giới hạn phần cứng Intel Mac x86_64, không phải lựa chọn) — mọi model
  cục bộ phải chạy qua ONNX Runtime, không qua `transformers`/`torch` trực tiếp.
