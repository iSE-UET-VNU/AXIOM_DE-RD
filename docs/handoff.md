# Bàn giao phiên làm việc — Retrieval Vật Lý ViDoRe V3

Đọc file này trước khi làm gì tiếp. Mục đích: người/phiên tiếp theo biết ngay đang đứng ở đâu,
không cần đọc lại toàn bộ lịch sử. Ledger kỹ thuật đầy đủ (số liệu, phương pháp, mọi refutation)
là [`docs/vidore_v3_results.md`](vidore_v3_results.md) — file này chỉ là bản đồ + nhật ký thời gian.

Cập nhật lần cuối: **27/08/2026, 21:15**.

---

## 1. Trạng thái git

- Branch: `feature/retrieval-baseline` (nhánh nghiên cứu — có đầy đủ ledger, script, cache refs).
- So với `origin/feature/retrieval-baseline`: các commit `00e8bdc / d79ca66 / 13b651a` + phiên
  27/08 (ledger §20–§21, `physics_stack.py`, `physics_kdl_arms.py`, `retrieval_research_plan.md`)
  **chưa push**. 0 behind.
- **Các nhánh khác đã check (27/08):**
  - `origin/on-demand-parsing` — pipeline light-preparation / on-demand PDF discovery (`research/data_discovery/`).
  - `origin/table_merge`, `origin/fix_table`, `origin/develop` — nơi bảng kết quả + TableAgent sống; state gọn hơn, **xoá phần lớn docs/** (kể cả ledger). Không merge ledger vào đó.
  - `feature/chunk-embed` — refactor chunking/embedding.
  - Code KDL + light-prep parsing **đã có sẵn trong cây hiện tại** (`src/ingestion/parsing/kdl_pdf_inspector.py`, `pdf_inspector.py`).
- `.claude/` untracked, không commit.
- `git push origin feature/retrieval-baseline` trước khi chạy Colab notebook dùng `git clone`
  (ColQwen2 notebook dùng `git archive` nên không bắt buộc).

---

## 2. Kết quả đã xác nhận

### 2a. Đo trên pool KDL production (recipe "Baseline Legacy": KDL → fixed_512/128 → MaxP → α0.7)

**Đây là các con số điền được vào bảng của team.** Baseline tái lập 43.86 / 46.73
(CSV ghi 44.2 / 47.47 — lệch ~0.35 do biến động run KDL). Ledger §21, `physics_kdl_arms.py`,
toàn bộ từ cache — không API, không GPU. Paired permutation 10k.

| Giải pháp | NDCG@10 | R@10 | Correct_only | Correct+partial | Δ NDCG | Chi phí |
|---|---:|---:|---:|---:|---:|---|
| Baseline Legacy (tái lập) | 43.86 | 46.73 | 50.17 | 91.03 | — | — |
| + SEP (λ=0.5) | 46.27 | 48.88 | 51.83 | 89.70 | +2.41 (p=.0004) | miễn phí, +0.3ms/câu |
| + ColQwen2 fusion (w≈0.6) | 47.47 | 49.77 | *chưa đo* | *chưa đo* | +3.61 (p<.001) | GPU Colab 1 lần |
| **+ SEP + ColQwen2 (w≈0.7)** | **48.35** | 50.63 | *chưa đo* | *chưa đo* | **+4.49** | như trên |
| + Voyage rerank-2.5 (trên KDL) | *CHƯA ĐO* | | | | (trên vidore_page: +5.08) | API ~100 phút, chặn ngân sách |

Light-prep: baseline 43.02 → SEP+ColQwen2 = **48.45** (+5.4). Chi tiết ledger §21.

### 2b. Đo trên `vidore_page` (text ViDoRe cung cấp, KHÔNG dùng parse của ta) — chỉ để so sánh

| Cải tiến | NDCG@10 | Δ vs 44.15 | p |
|---|---:|---:|---:|
| SEP | 46.17 | +2.02 | 0.0026 |
| Voyage rerank-2.5 depth-20 | **49.23** | **+5.08** | 0.0001 |
| ColQwen2 fusion w≈0.7 | 47.37 | +3.22 | 0.0014 |
| **stack: Voyage + ColQwen2** | 47.5–49.8 | **n.s. vs Voyage một mình** | — |

**Bài học stacking (ledger §20–§21):**
- **CÓ reranker:** SEP và ColQwen2 bị Voyage hấp thụ hết — stack không vượt Voyage một mình (49.23). Ba lever sửa cùng một lỗi (§7).
- **KHÔNG reranker (pool KDL):** SEP + ColQwen2 CÓ stack → 48.35, cách Voyage-một-mình ~1 điểm, hoàn toàn miễn phí.
- Generator swap gpt-5.2: Correct 71.67% (+27.73pp, p=.0001) ở n=120/302 — QA, không phải NDCG; chặn ngân sách OpenRouter.

Mốc tham khảo: SOTA ViDoRe V3 physics = **50.84** (nemotron-colembed-8b, trung bình 6 ngôn ngữ; ladder ta chỉ Pháp).
Kế hoạch: [`docs/retrieval_research_plan.md`](retrieval_research_plan.md).

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
| `physics_stack.py` (stack 3 lever, vidore_page) | 27/08 ~15:05 | 27/08 ~15:08 | **~3 phút** | toàn bộ từ cache; 10k permutation × ~10 arm |
| `physics_kdl_arms.py` (SEP+ColQwen2 trên KDL + light-prep) | 27/08 ~20:55 | 27/08 ~21:02 | **~7 phút** | build 2 pool chunk-level từ cache embedding + 24 arm × 10k permutation |

**Bài học rút ra, nên áp dụng tiếp:** hai lần chạy dài nhất phiên này (rerank Voyage 100 phút,
rerank cục bộ 3h31m) đều là **gọi API bị giới hạn tốc độ** hoặc **suy luận CPU không có GPU** —
cả hai đều là hệ quả trực tiếp của ràng buộc "chỉ có Colab cho GPU, key OpenRouter có hạn mức".
Không có thí nghiệm nào trong phiên này chạy lâu vì thuật toán chậm.

---

## 5. Việc cần làm tiếp — theo thứ tự ưu tiên

Kế hoạch đầy đủ + lập luận: [`docs/retrieval_research_plan.md`](retrieval_research_plan.md).

0. **Push commit lên origin** — 1 dòng lệnh.
1. **Điền bảng team bằng số §2a** (KDL). Voyage-trên-KDL còn thiếu: chạy `physics_rerank_voyage.py
   --pool data/benchmark/vidore_v3/results/physics_KDL_pool.json --texts <kdl_page_texts>`
   (~100 phút API — **hỏi ngân sách trước**). `physics_kdl_arms.py` đã dump sẵn pool.
2. **P1 — nâng visual arm ColQwen2-2B → Nemotron ColEmbed V2 4B/3B.** Kỳ vọng lợi lớn nhất.
   Check gating + khả thi Colab T4 trước. Dùng lại pattern `ColQwen2_visual_arm_physics.ipynb`.
3. **P3 — reranker mạnh hơn: Qwen3-Reranker-4B trên Colab GPU** (depth 50/100, không cắt 1200 ký tự).
   `bge-reranker-base` đã bị loại (§19) — **đo lại từ đầu**, đừng giả định thắng Voyage.
4. **P2 — leg dense: `text-embedding-3-small` → Qwen3-Embedding.** Leg dense (41.1 một mình) là mắt xích yếu nhất.
5. **P4/P4b — α thích ứng theo truy vấn** (trần +8.60, §12). Hai hướng: cross-encoder làm router (P4),
   hoặc LambdaMART/Metarank làm tầng fusion học được (P4b — xem plan; train `lightgbm.LGBMRanker`
   thẳng, Metarank chỉ là serving layer).
6. **Generator swap gpt-5.2, đủ 302 câu** — +27.73pp ở n=120, chặn ngân sách OpenRouter.

**Với MỌI component mới: đo paired vs Voyage-một-mình (49.23), không chỉ vs baseline** — §20/§21
cho thấy delta vs baseline gây hiểu nhầm vì các lever trùng lặp.

---

## 6. Vị trí file quan trọng

| Loại | File |
|---|---|
| Ledger kỹ thuật đầy đủ (21 mục) | `docs/vidore_v3_results.md` |
| Kế hoạch nghiên cứu tiếp (paper review + P1–P5 + Metarank) | `docs/retrieval_research_plan.md` |
| Giải thích SEP tiếng Việt | `docs/sep_giai_thich.md` |
| Phân tích first-principles tiếng Việt | `docs/phan_tich_first_principles.md` |
| Notebook Colab ColQwen2 (template cho P1) | `ColQwen2_visual_arm_physics.ipynb` |
| Stack 3 lever trên vidore_page | `research/experiments/physics_stack.py` |
| SEP + ColQwen2 trên KDL / light-prep (số điền bảng) | `research/experiments/physics_kdl_arms.py` |
| Eval ColQwen2 (đọc export Colab, không GPU) | `research/experiments/physics_colqwen_eval.py` |
| Reproduce baseline KDL + SEP | `research/experiments/vidore_prod_baseline_sep.py` |
| Pool KDL / light-prep đã dump (cho Voyage rerun) | `data/benchmark/vidore_v3/results/physics_KDL_pool.json`, `physics_light_prep_pool.json` |
| Artifact báo cáo cho trưởng nhóm (cần cập nhật đến §21) | https://claude.ai/code/artifact/bd820616-d28c-4813-ae82-0a7351204a01 |

---

## 7. Ràng buộc đang đứng — không tự ý vượt qua

- **Hết ngân sách OpenRouter** (key giáo viên, hết hạn mức tháng) — không tự ý chạy thêm gì cần
  API mà chưa hỏi lại.
- GPU chỉ có qua Google Colab (không có CUDA/MPS cục bộ) — mọi arm cần GPU phải đóng gói thành
  notebook, theo đúng pattern `git archive` + upload zip (không dùng `git clone` vì repo private).
- torch cục bộ pin ở 2.2.2 (giới hạn phần cứng Intel Mac x86_64, không phải lựa chọn) — mọi model
  cục bộ phải chạy qua ONNX Runtime, không qua `transformers`/`torch` trực tiếp.
