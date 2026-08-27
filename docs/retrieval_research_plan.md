# Kế hoạch nghiên cứu — nâng NDCG@10 retrieval (physics, French)

Viết ngày **27/08/2026**. Nối tiếp [`handoff.md`](handoff.md) và ledger
[`vidore_v3_results.md`](vidore_v3_results.md) §20 (đo stacking).

---

## 1. Đặt lại mục tiêu — con số 60 nằm ở đâu

| Mốc | NDCG@10 physics | Nguồn |
|---|---:|---|
| BM25S (bản công bố) | 39.8 | paper ViDoRe V3 Table 10 |
| Fusion α=0.7 hiện tại của ta | 44.15 | ledger §1b |
| Pipeline physics tốt nhất trong paper | 48.2 | Jina-v4 + zerank-2, paper Table 2 |
| **SOTA leaderboard hiện tại** (Feb 2026) | **50.84** | Nemotron ColEmbed V2 8B, [arXiv 2602.03992](https://arxiv.org/abs/2602.03992) Table 2 |
| Trần khi stack 3 lever đã có | **49.23** | ledger §20, đo hôm nay (= Voyage rerank một mình) |

**60 cao hơn ~10 điểm so với model mạnh nhất thế giới trên domain này.** Physics
là một trong các domain khó nhất của ViDoRe V3. Lưu ý: 50.84 là **trung bình 6
ngôn ngữ query**, còn ladder của ta **chỉ tiếng Pháp** — nên đây là mốc tham
khảo gần đúng, không phải so sánh 1-1 (xem ledger §4). Harness của ta đo được
BM25S ở 40.15 vs bản công bố 39.8 (ledger §1a) → thang đo *của ta* đúng; điều đó
không nói gì về khớp ngôn ngữ với dòng leaderboard.

**Mục tiêu thực tế:** bằng hoặc nhỉnh hơn SOTA physics (~50), tức mọi điểm trên
49.23 đều là tiến bộ thật. Không chốt một con số cụ thể — ledger §11 và §20 đều
cấm cộng dồn Δ, và §20 vừa cho bằng chứng trực tiếp rằng các lever compose
dưới-cộng-tính (sub-additive). Đánh giá theo *component*, không theo tổng Δ dự
phóng.

---

## 2. Kết quả đo hôm nay — stack 3 lever đã có (ledger §20)

Chạy `physics_stack.py`, cùng một pool `vidore_page` α=0.7, cùng qrels, paired
permutation. Fusion báo cáo theo band `w`, không lấy argmax.

| arm | NDCG@10 | so với | kết luận |
|---|---:|---|---|
| baseline α=0.7 | 44.15 | — | |
| SEP (λ=0.5) | 46.17 | baseline +2.02 | có ý nghĩa |
| ColQwen2 fusion (w 0.2–0.8) | 44.7–47.5 | baseline, tối đa +3.38 | có ý nghĩa ở w≥0.4 |
| **Voyage rerank-2.5 top-20** | **49.23** | baseline +5.07 | **trần của bộ công cụ hiện tại** |
| SEP + ColQwen2 (w 0.2–0.8) | 47.1–48.3 | baseline +2.9…+4.1 | vẫn dưới Voyage một mình |
| **Voyage + ColQwen2 (w 0.2–0.8)** | **47.5–49.8** | **Voyage: +0.6 … −1.8** | **n.s. ở MỌI w** |

**Không gì stack được lên reranker.** Voyage rerank một mình = 49.23. Fuse
ColQwen2 lên trên **không** vượt nó có ý nghĩa thống kê (đỉnh +0.61 tại w=0.3,
p=0.34; âm khi w≥0.4). Ba lever đều sửa cùng một lỗi (ledger §7); sau khi
cross-encoder đã sắp lại top-20 theo tương tác token, structural prior và visual
arm ColQwen2-2B không còn gì để thêm.

Tín hiệu đáng chú ý: **trọng số visual hữu ích sụp đổ khi có Voyage** — đỉnh w
dịch 0.7 (trên fusion thô, +3.38) → 0.3 (trên thứ tự đã rerank, +0.61 n.s.).
Visual arm chủ yếu đang phục hồi thứ tự mà reranker phục hồi tốt hơn.

**SEP sau rerank không đo sạch được từ cache** (SEP cần điểm số thật, không phải
rank proxy). Hướng đã rõ: SEP không giúp sau reranker.

→ **Kết luận: hết đường với các lever hiện tại KHI ĐÃ CÓ RERANKER. Cần component
mạnh hơn.**

### 2b. Cùng lever, đo lại trên pool KDL production (ledger §21) — điền bảng được

Bảng ở trên đo trên `vidore_page`. Bảng của team bắt buộc KDL. Đo lại trên
**KDL → fixed_512/128 → MaxP → α=0.7** (đúng recipe "Baseline Legacy"), toàn bộ
từ cache — không API, không GPU:

| arm | NDCG@10 | R@10 | Δ vs 43.86 |
|---|---:|---:|---:|
| Baseline Legacy (tái lập) | 43.86 | 46.73 | — |
| + SEP (λ=0.5) | 46.27 | 48.88 | +2.41 (p=0.0004) |
| + ColQwen2 fusion (w≈0.6) | 47.47 | 49.77 | +3.61 (p<0.001) |
| **+ SEP + ColQwen2 (w≈0.7)** | **48.35** | 50.63 | **+4.49** |

Light-prep tương tự: 43.02 → SEP+ColQwen2 **48.45** (+5.4). **Khác §2a: ở đây
KHÔNG có reranker → SEP + ColQwen2 CÓ stack (+2 trên SEP, p≈0.02).** Stack miễn
phí này (~48.3) cách Voyage-một-mình (49.23) chỉ ~1 điểm.
Voyage trên pool KDL: **chưa đo** — cần chạy API ~100 phút (đang chặn ngân sách).

---

## 3. Đọc paper đính kèm — "Spatially-Grounded Document Retrieval" (Snappy)

Paper này về **định vị vùng trong trang** (IoU với bounding box bằng chứng), chấm
bằng BBox-DocVQA. **§6 của nó nói rõ page-level retrieval nằm ngoài phạm vi.** Nó
không trực tiếp nâng NDCG@10. Hai điều chuyển giao được:

1. **Quy mô model visual (§6.2, §6.5).** ColQwen3-4B ≫ ColModernVBERT-250M (chênh
   14.2 pp ở IoU@0.5), nhưng ColQwen3-8B ≈ 4B (59.8 vs 59.7 — bão hòa). Củng cố
   hướng: **nâng visual arm lên lớp 4B, đừng trả tiền cho 8B.** Arm hiện tại là
   ColQwen2-**2B**.
2. **Tách hai giai đoạn (§3.4).** Mean-pool patch → ANN lấy ứng viên; MaxSim đầy
   đủ → rerank trên ứng viên. Ledger §18 ghi "ColQwen2 dùng làm reranker chưa
   test" — ta mới chỉ dùng nó làm arm fusion. Đây là một thí nghiệm riêng trên
   score đã có.

DeepSeek-OCR + gán vùng của Snappy là chuyện **giảm token phía generation**, không
phải retrieval — bỏ qua ở đây.

> Ghi chú: bạn viết "các bài báo sau" (số nhiều) nhưng chỉ có 1 PDF đính kèm. Nếu
> có paper khác muốn tôi đọc, gửi lại.

---

## 4. Related work đã tra (Aug 2026)

| Model / kỹ thuật | Liên quan | Nguồn |
|---|---|---|
| **Nemotron ColEmbed V2** (3B/4B/8B, open weights, Qwen3-VL backbone) | SOTA ViDoRe V3, physics 50.84. Bản 4B trên HF. Drop-in thay ColQwen2 làm visual arm. | [arXiv 2602.03992](https://arxiv.org/abs/2602.03992), [HF](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) |
| **Qwen3-VL-Embedding / Qwen3-VL-Reranker** | Framework hợp nhất: vừa là dense arm đa phương thức vừa là reranker. | [arXiv 2601.04720](https://arxiv.org/pdf/2601.04720) |
| **Qwen3-Reranker** (0.6B/4B/8B, Apache 2.0, 100+ ngôn ngữ, 32k context) | Reranker open mạnh — `bge-reranker-base` (đã bị loại, ledger §19) yếu hơn hẳn. Chạy trên Colab GPU để thoát trần depth-20 + cắt 1200 ký tự của Voyage free-tier. | [Qwen blog](https://qwenlm.github.io/blog/qwen3-embedding/), [HF](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| **Qwen3-Embedding** (text, 0.6B/4B/8B) | Thay `text-embedding-3-small` cho leg dense văn bản. Paper: Qwen3-0.6B đơn lẻ đã 43.8 vs fusion của ta 44.1. | như trên |
| **Visual RAG Toolkit** — training-free pooling + multi-stage search | Làm rẻ arm multi-vector kiểu ColQwen (nén số vector/trang). | [arXiv 2602.12510](https://arxiv.org/pdf/2602.12510) |
| **Metarank** — LambdaMART LTR service, self-host | **KHÔNG phải cross-encoder ngữ nghĩa** — là GBDT xếp hạng theo *feature*, cần dữ liệu huấn luyện (qrels/click). Xem P4b. | [github](https://github.com/metarank/metarank), [docs](https://docs.metarank.ai/) |

---

## 5. Thí nghiệm tiếp theo — theo thứ tự ưu tiên

Ràng buộc chi phối thứ tự (từ handoff §7): GPU chỉ qua Colab; hết ngân sách
OpenRouter; torch cục bộ pin 2.2.2 (chỉ ONNX).

### P1 — Nâng visual arm: ColQwen2-2B → lớp 4B *(kỳ vọng lợi lớn nhất)*

- **Model:** `nvidia/nemotron-colembed-vl-4b-v2` (ưu tiên — SOTA), hoặc một
  ColQwen3-4B. **Trước khi dựng notebook, check 2 thứ:** (a) license/gating trên
  HF; (b) khả thi trên Colab T4 — backbone Qwen3-VL-4B multi-vector trên 1674
  trang, T4 16GB không bf16 có thể là blocker cứng; nếu vậy cần A100 (Colab Pro)
  hoặc bản 3B (`nemotron-colembed-vl-3b-v2`, avg 59.8).
- **Cách làm:** đúng pattern `ColQwen2_visual_arm_physics.ipynb` — Colab GPU,
  export ma trận điểm 302×1674 phẳng, chấm cục bộ bằng `physics_colqwen_eval.py`
  (không GPU/API). Gần như chỉ đổi model id + kích thước ảnh render.
- **Đo:** visual-only NDCG@10, fusion sweep w, complementarity vs text (như §18).
  Rồi cắm vào `physics_stack.py` thay ColQwen2.
- **Vì sao P1:** visual arm là nơi paper Snappy + leaderboard đều chỉ ra còn
  nhiều dư địa; ColQwen2-2B alone chỉ 45.7 (n.s.), Nemotron-class có thể tự nó
  vào high-40s/low-50s.

### P2 — Thay leg dense văn bản: `text-embedding-3-small` → Qwen3-Embedding-4B

- Leg dense hiện tại một mình chỉ 41.1; nó kéo cả fusion xuống.
- **Cách làm:** re-embed 1674 trang + 302 query (rẻ). Nếu chạy được qua ONNX cục
  bộ thì tốt; nếu không, Colab một lần rồi cache vector.
- **Đo:** dense-only, rồi α sweep lại (α tối ưu sẽ đổi khi leg dense mạnh lên),
  rồi fusion.
- Chưa có trong handoff §5 — đây là lever chưa đụng tới, có thể lớn hơn mọi thứ
  trong danh sách cũ.

### P3 — Reranker mạnh hơn: Qwen3-Reranker-4B trên Colab GPU

- Ledger §19: bài học là "cross-encoder yếu làm hại", **không** phải "rerank vô
  dụng" — Voyage +5.08 là bằng chứng đối chứng. `bge-reranker-base` (278M) quá
  yếu; Qwen3-Reranker-4B là lớp khác.
- **Lợi thế so với Voyage:** thoát trần free-tier depth-20 (thử depth 50/100) và
  thoát cắt 1200 ký tự (ledger §1b-iii nói +5.08 là *sàn* dưới reranker không bị
  cắt).
- **Đo lại từ đầu**, đừng giả định thắng Voyage. Cắm vào `physics_stack.py`.

### P4 — Query-adaptive α bằng cross-encoder làm router

- Ledger §12: trần oracle per-query α là **+8.60** — lớn nhất chưa khai thác.
  §14: trần DAT +3.27 một mình, +5.01 với SEP. §15: LLM sinh làm giám khảo chạy ở
  mức ngẫu nhiên (50.2% vs base rate 35.3%) — **§15 đã chỉ đích danh cách sửa:
  dùng cross-encoder relevance model, không phải LLM sinh.**
- Giờ đã có ứng viên cross-encoder (P3). Cho nó chấm top-1 của mỗi leg → chuẩn hóa
  thành α. Đây là con đường duy nhất còn sống tới +8.60.

### P4b — LambdaMART LTR làm tầng fusion học được (Metarank / LightGBM)

**Metarank là gì:** dịch vụ rerank thứ cấp dùng **LambdaMART (GBDT)** trên
feature engineering pipeline (YAML DSL), self-host được (standalone in-memory
hoặc Redis), độ trễ rerank 10–20ms, huấn luyện từ *ranking + nhãn liên quan*
(qrels hoặc click). [github.com/metarank/metarank](https://github.com/metarank/metarank).

**Không phải cái thay thế Voyage.** Metarank xếp hạng theo *feature*, không đọc
sâu văn bản như cross-encoder → không bắt được ca "trang gold khớp chữ ít nhưng
đúng ngữ nghĩa" (đúng chỗ Voyage +5.08 thắng). Nó là **tầng fusion học được**,
thay cho α=0.7 + β SEP + w ColQwen2 chỉnh tay — và điểm cross-encoder (P3) nên
là *một feature* đầu vào của nó, không phải đối thủ.

**Vì sao đáng thử:** ledger §12 — trần per-query α là **+8.60**, 7 predictor
tuyến tính đều r<0.13. GBDT trên feature *tương tác câu hỏi–corpus* là công cụ
đúng để bắt tín hiệu query-conditional đó. Features có sẵn cho mỗi (query, page):
điểm & rank BM25/dense/RRF/α-fuse, A_file & N của SEP, điểm & rank ColQwen2, cờ
cùng-file-với-anchor, kề anchor, độ dài trang/truy vấn, số từ truy vấn, (khi có)
điểm cross-encoder.

**Cách làm (nghiên cứu trước, Metarank sau):**
- Giai đoạn nghiên cứu: **train `lightgbm.LGBMRanker` (objective lambdarank)
  thẳng trong Python** — không phụ thuộc torch, hợp ràng buộc 2.2.2. k-fold theo
  query (302 physics ít → gộp pharma/hr/cs nếu feature chuyển giao), cây nhỏ,
  regularize mạnh. Đo vs baseline **và vs SEP+ColQwen2 stack** (§21).
- Chỉ khi thắng: đóng gói Metarank làm tầng serving (Docker, standalone). Serving
  layer của Metarank là thừa với giai đoạn nghiên cứu.
- **Rủi ro:** 302 query nhỏ cho LTR; gold dày (57% trang là gold cho *câu nào đó*)
  → phân bố nhãn lạ, dễ overfit. Ưu tiên P1/P3 trước.

### P5 — ColQwen làm reranker thay vì arm fusion (từ Snappy §3.4)

- Rẻ, dùng ma trận điểm đã có: xếp lại top-20/50 của pool text bằng MaxSim visual
  thay vì blend điểm toàn cục. Kiểm tra xem late-interaction visual có sửa
  within-file ordering tốt hơn blend không.

### Không làm lại

- SEP đứng sau reranker (ledger §20: không đo sạch được, hướng đã rõ là không giúp).
- Stack SEP hoặc ColQwen2-2B vào pipeline đã có reranker mạnh — redundant (§20).
- `bge-reranker-base` hay reranker <300M nói chung (§19).
- Bất cứ thứ gì "localise" trong trang: DCW, sub-page MaxSim, boost theo block
  type (ledger §8, §9 — "localise → thua, aggregate → thắng").

---

## 6. Lập luận theo component (KHÔNG dự phóng tổng)

Ledger §11 và §20 cấm cộng dồn Δ, và §20 vừa chứng minh các lever compose
dưới-cộng-tính. Không viết ra một con số mục tiêu. Đánh giá từng component:

1. **P1 (visual 4B)** — arm visual hiện 45.7 (n.s. một mình). Snappy §6 +
   leaderboard đều chỉ ra đây là chỗ còn dư địa lớn nhất. Một model lớp Nemotron
   có thể tự nó đạt low-50s. Đây là thay-thế-component, không phải stack.
2. **P3 (reranker mạnh, depth cao)** — Voyage +5.07 ở depth-20 bị cắt 1200 ký
   tự là *sàn* (ledger §1b-iii). Reranker là lever mạnh nhất và là thứ duy nhất
   không bị hấp thụ (§20). Một reranker mạnh hơn ở depth cao hơn là đường nâng
   trần thẳng nhất.
3. **P2 (dense 4B)** — nâng sàn của fusion trước khi rerank; ảnh hưởng gián tiếp
   nhưng leg dense hiện là mắt xích yếu nhất (41.1 một mình).
4. **P4 (adaptive α)** — cửa duy nhất chạm phần lớn của trần +8.60 (§12).

Thứ tự kỳ vọng-lợi: P1 ≈ P3 > P2 > P4. Mỗi cái đo riêng, cắm vào `physics_stack.py`,
và **đo paired vs Voyage-một-mình** (không chỉ vs baseline) — đó là bài test
quyết định như §20.

---

## 7. Việc làm ngay

1. `git push origin feature/retrieval-baseline` (các commit chưa push + §20 + `physics_stack.py`).
2. P1: kiểm tra gating + khả thi T4 cho `nvidia/nemotron-colembed-vl-4b-v2` (hoặc 3B), dựng
   notebook từ `ColQwen2_visual_arm_physics.ipynb`.
3. P2 song song: script re-embed với Qwen3-Embedding (không phụ thuộc GPU nếu ONNX chạy).
