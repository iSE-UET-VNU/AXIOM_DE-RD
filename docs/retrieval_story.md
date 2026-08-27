# Câu chuyện nghiên cứu Retrieval — ViDoRe V3 Physics (French)

Tài liệu này kể lại **toàn bộ luồng suy nghĩ**: tại sao chạy từng thí nghiệm, tại
sao chọn từng công cụ, thu được gì, rút ra điều gì, và đi tiếp đâu. Số liệu chi
tiết + p-value ở [`vidore_v3_results.md`](vidore_v3_results.md); kế hoạch ở
[`retrieval_research_plan.md`](retrieval_research_plan.md); file này là mạch truyện.

Cập nhật: **27/08/2026**.

---

## 0. Bài toán

Retrieval trên ViDoRe V3, subset **physics, tiếng Pháp**: 302 câu hỏi, 1.674
trang, 42 tài liệu, tìm trong "lake-subset". Chỉ số chính **NDCG@10**. Baseline
pipeline (KDL parse → chunk 512 → embed `text-embedding-3-small` → hybrid
BM25+dense α=0.7) đạt **~44**. Câu hỏi: nâng lên bao nhiêu, bằng cách nào.

---

## 1. Trước tiên: cái thước đo có đúng không?

**Tại sao:** không thể tối ưu một con số nếu chưa chắc con số đó đo đúng.

**Làm gì:** cài đúng thư viện `bm25s` của chính paper, chạy trên corpus / query /
qrels / hàm NDCG của ta.

**Kết quả:** ta ra 40.15, paper công bố 39.8 — **khớp**. Vậy corpus, bộ lọc ngôn
ngữ, phép join page-id, và cách tính NDCG của ta đều đúng. Khoảng cách −2.7 của
BM25 ta so với BM25S là **hoàn toàn do tokenization** (tiếng Pháp có elision:
`l'énergie` bị dính thành 1 token nên query `énergie` không khớp).

**Rút ra:** thước đo chuẩn. Mọi Δ sau này là thật, không phải lỗi đo. Đây là
bước bị bỏ qua nhiều nhất và tốn kém nhất nếu sai.

---

## 2. Chẩn đoán: điểm mất nằm ở đâu?

**Tại sao:** trước khi thử giải pháp, phải biết lỗi có hình dạng gì.

**Làm gì:** phân tích recall theo độ sâu, và phân loại các câu fail.

**Kết quả:**
- **Recall gần bão hòa**: any-gold@100 = 98.3%. Trang gold gần như luôn nằm trong
  pool → đây là bài toán **xếp thứ tự**, không phải bài toán **tìm kiếm**.
- Phân loại fail: **69%** là "file đúng đã nằm trong top-10, nhưng sai trang bên
  trong file đó"; 7% là "file gold không có trong top-10"; 24% là "gold đã đủ
  trong top-10 rồi".
- Cùng-file = trang gold có xác suất cao **16.9×**; ứng viên khác-file chỉ là
  gold 0.46% số lần → gần như thuần nhiễu.
- **Trần lý thuyết của α thích ứng theo câu**: nếu mỗi câu được chọn α tối ưu
  riêng → +8.60 NDCG (lớn hơn mọi thứ khác). Nhưng 7 đặc trưng bề mặt câu hỏi
  đều r < 0.13 → không có predictor rẻ.

**Rút ra:** vấn đề là **phân biệt trang trong cùng một file** + **fusion phụ
thuộc câu hỏi**. Corpus này **topical, nhỏ, gold phân tán và dư thừa** (7.21
trang gold/câu, ~2 trang là đủ trả lời) — **KHÔNG phải** needle-in-haystack.

---

## 3. Nhóm thí nghiệm A — "khu trú tín hiệu" (đều thất bại)

**Giả thuyết chung:** mỗi trang trong 1 tài liệu chia sẻ từ vựng chủ đề → thành
phần "chủ đề tài liệu" lấn át, làm các trang giống nhau. Loại bỏ nó đi thì phân
biệt được trang.

| Thí nghiệm | Ý tưởng | Kết quả | Bài học |
|---|---|---|---|
| **DCW** (trừ centroid tài liệu) | `e_p − κ·μ_f` | **−5.7 / −9.5**, sai dấu, tệ hơn ở tài liệu lớn | Thành phần "chủ đề tài liệu" là **tín hiệu, không phải nhiễu** — query ViDoRe là topical, phần lớn "trang này có liên quan không" = "tài liệu này có nói về chủ đề này không" |
| **MaxSim dưới cấp trang** | chunk nhỏ hơn, MaxSim gộp về trang | **−1.2 → −11.2**, đơn điệu theo độ mịn | Phân mảnh mất 5+ điểm. MaxSim chỉ hoạt động khi model **được huấn luyện** với mục tiêu MaxSim (ColBERT/ColPali), không phải gắn vào một embedder pooled |
| **Trọng số theo loại block** | table/figure/equation nặng hơn | không loại nào giàu gold hơn (tỉ lệ 0.80–1.03) | Đừng cài boost dựa trên trực giác |
| Union hai nhánh cho reranker | ngừng vứt ứng viên của nhánh thua | **−5.7** ở cùng budget | Fusion chấm mọi item bằng **cả hai** tín hiệu; union chỉ ghép hai góc nhìn cục bộ |
| Lan truyền multi-hop | trang liên quan kéo trang liên quan | **−2.0 / −6.6** | Khuếch tán nhiễu |

**Mẫu hình xuyên suốt nhóm A: khu trú → thua. Gộp → thắng.**

---

## 4. Nhóm B — "gộp tín hiệu": SEP (thành công, có điều kiện)

**Tại sao:** nhóm A cho thấy nên đi hướng ngược lại — khuếch đại cấu trúc, không
bóc tách nó.

**Ý tưởng:** DISRetrieval (arXiv 2506.06313) dựng cây discourse rồi cho node
liên quan "đề bạt" các lá con của nó. Ablation của chính nó nói: nửa đắt tiền
(node tóm tắt bằng LLM) là nửa **nên bỏ**. Phần còn lại là gộp-và-đề-bạt, chỉ
cần một cái cây — mà corpus của ta **đã có sẵn**: `file → page` nằm ngay trong
unit-id.

**Cơ chế (SEP — Structural Evidence Propagation):**
```
s'(c) = λ·s(c) + (1−λ)·[ β·A_file(c) + (1−β)·N(c) ]
```
`A_file` = trung bình top-m điểm pool của file; `N` = bằng chứng từ trang lân
cận, giảm theo khoảng cách. Chỉ sắp xếp lại pool → recall@100 không đổi. Không
API, không GPU, +0.3ms/câu.

**Kết quả:** **+2.41 NDCG@10** (p=0.0004) trên pool KDL production. Chuyển giao
sang **pharmaceuticals: +2.17** (p=0.0005) — đây là phép thử out-of-domain quan
trọng nhất (khác subset, khác domain, config y nguyên).

**Điều kiện:** SEP cần bằng chứng cấp-file để ràng buộc *trang nào* liên quan →
chỉ đúng khi tài liệu nhỏ. Dự đoán **đăng ký trước khi test**, đúng **5/5**:

| subset | trang/file | gold % của file | dự đoán | đo được |
|---|---|---|---|---|
| physics | 39.9 | 14.6% | giúp | +2.02 ✓ |
| pharmaceuticals | 44.5 | 12.0% | giúp | +2.17 ✓ |
| hr | 79.3 | 9.5% | yếu | +0.54 n.s. ✓ |
| industrial | 194.2 | 5.2% | fail | −0.14 ✓ |
| computer_science | 680.0 | 0.8% | fail nặng nhất | −0.09 ✓ |

**Rút ra:** một prior cấu trúc rẻ giúp **đúng khi tài liệu cỡ một chương sách**
(gold ≳10% của file). Có một bài screen miễn phí để biết trước.

---

## 5. Nhóm C — fusion thích ứng câu hỏi (DAT): trần đúng, hiện thực hóa thất bại

**Tại sao:** §2 tìm ra +8.60 headroom thật của α-per-query, 7 đặc trưng bề mặt
câu hỏi đều không bắt được. Paper DAT (arXiv 2503.23013) nói vì sao: tín hiệu
không nằm ở **câu hỏi**, nó nằm ở **kết quả** — chấm top-1 của mỗi nhánh xem
nhánh nào làm tốt hơn.

**Kết quả:**
- **Trần với giám khảo hoàn hảo** (dùng qrels thật): +3.27 một mình, **+5.01 khi
  cộng với SEP** — hai cái bổ sung nhau.
- **Với LLM giám khảo thật** (gpt-4o): **thất bại**. Độ chính xác 50.2% so với
  base rate 35.3% → **ở mức ngẫu nhiên** trên văn xuôi vật lý tiếng Pháp. Giám
  khảo nói "có liên quan" cho 77% trường hợp → verdict gần như vô nghĩa.

**Rút ra:** giám khảo phải là **model relevance được huấn luyện** (cross-encoder),
không phải LLM sinh chung chung. Trần +5.01 vẫn còn giá trị, chờ giám khảo tốt
hơn.

---

## 6. Nhóm D — reranking: người thắng cuộc

**Tại sao:** §2 nói pages đã ở trong pool, chỉ xếp sai. Đó chính xác là việc của
cross-encoder — nó chấm tương tác token giữa câu hỏi và từng trang, thứ mà
bi-encoder pooled không làm được.

| Thí nghiệm | Kết quả | Bài học |
|---|---|---|
| **Voyage rerank-2.5, depth-20** | **+5.08** (p=0.0001) — Δ lớn nhất đo được | Chẩn đoán §2 đúng: sắp lại top-20 là đủ |
| Rerank cục bộ `bge-reranker-base` (278M, ONNX, miễn phí) | **−2.9 → −6.6**, tệ hơn theo độ sâu | "Có huấn luyện" ≠ "tốt". Cross-encoder **yếu làm hại nhiều hơn không làm gì** — nó đề bạt false positive mà fusion đã loại đúng |

**Rút ra:** phương pháp khu trú **duy nhất thắng** là một cross-encoder
**được huấn luyện** cho tương tác query–document. Bi-encoder geometry (nhóm A)
không làm được việc này; cross-encoder làm được. Nhưng phải đủ mạnh.

---

## 7. Nhóm E — visual: từ vô dụng đến bổ sung thật

| Thí nghiệm | Kết quả | Bài học |
|---|---|---|
| **CLIP ViT-B/32** (224px, encoder ảnh tự nhiên) | 4.45 vs text 43.86; fusion không giúp ở mọi trọng số | Encoder ảnh tự nhiên ở 224px không đọc được trang chữ dày |
| **ColQwen2** (Qwen2-VL-2B, late-interaction, doc-VLM thật) | fusion **+3.22** (p=0.0014) | Arm visual đầu tiên có ý nghĩa. Bổ sung thật: 7.9% gold chỉ-visual tìm được vs 7.5% chỉ-text — không phải tiếng vọng của text |

**Rút ra:** visual có giúp, nhưng chỉ với **document-VLM ở độ phân giải tài
liệu**. Model và độ phân giải quan trọng hơn "có dùng visual hay không".

---

## 8. Nhóm F — generator: bài học phương pháp luận đắt giá nhất

**Tại sao:** sau khi SEP cho +2.41 NDCG nhưng **không** làm QA nhúc nhích, câu
hỏi hiển nhiên (hỏi quá muộn): *cái gì* mới làm QA nhúc nhích?

**Làm gì:** giữ retrieval **byte-identical**, chỉ đổi generator, n=120 câu.

**Kết quả:** DeepSeek-V4-Flash → **gpt-5.2**: Correct_only **+27.73pp**
(p=0.0001) — **~20× mọi thay đổi retrieval**. gpt-5.2 đạt 71.67%, đúng bằng con
số 71.2% của paper (Gemini 3 Pro).

**Rút ra (áp dụng cho mọi dự án):** **đo trần của từng component trước khi tối ưu
bất kỳ cái nào.** Retrieval có ~3.6pp headroom E2E và ngốn cả session; generation
có ~27pp và chỉ tốn 1 thí nghiệm. Bằng chứng cho việc này đã nằm sẵn trong ledger
từ đầu (oracle-retrieval QA ceiling 55.6% với gpt-4o-mini vs paper 71.2%) — chỉ
là không ai đọc nó đúng lúc.

---

## 9. Chúng có cộng lại được không? (câu hỏi lớn nhất còn lại)

Ba lever xác nhận (SEP, Voyage, ColQwen2) mỗi cái đo trên baseline hơi khác nhau,
**chưa từng đo cùng lúc**. `physics_stack.py` + `physics_kdl_arms.py` giải quyết.

**Trên `vidore_page`, CÓ reranker:**
- Voyage một mình = **49.23**. SEP và ColQwen2 fuse lên trên **không vượt có ý
  nghĩa** (n.s. ở mọi trọng số). Trọng số visual hữu ích sụp từ 0.7 → 0.3.
- **Ba lever trùng lặp** — đều sửa cùng lỗi §2. Cross-encoder đã sắp lại top-20
  bằng tương tác token → structural prior và visual arm không còn gì để sửa.

**Trên pool KDL production, KHÔNG reranker:**
- SEP + ColQwen2 **CÓ** cộng: 43.86 → **48.35** (+4.5), hoàn toàn miễn phí (1
  lần GPU cho index visual).
- Cách Voyage-một-mình chỉ ~1 điểm.

**Rút ra:** khi chưa có reranker mạnh, các prior rẻ cộng dồn có ích. Khi đã có
reranker mạnh, chúng thành thừa. Kiến trúc thắng cuộc **không phải** chồng nhiều
lever yếu — mà là **một retriever tốt + một cross-encoder mạnh**.

---

## 10. Thí nghiệm cuối: reranker đa phương thức miễn phí

**Tại sao:** (1) §21 để hở "Voyage trên pool KDL = chưa đo"; (2) OpenRouter vừa
có `nvidia/llama-nemotron-rerank-vl-1b-v2:free` — cross-encoder 1.7B **đa phương
thức** (nhận cả ảnh trang), **miễn phí**, không dính trần free-tier 3 RPM của
Voyage. Về lý thuyết nó thay được **cả Voyage lẫn ColQwen2** trong một lời gọi.

**Làm gì:** rerank top-20 pool KDL production, text-only, depth 20 khớp Voyage.
302 lời gọi trong **~10 phút, chi phí 0, không đụng trần rate-limit nào** (khác
hẳn Voyage 100 phút vì 3 RPM).

**Kết quả (ledger §22):**

| arm | NDCG@10 | Δ | p |
|---|---:|---:|---:|
| baseline KDL α=0.7 | 43.86 | — | |
| **+ Nemotron rerank** | **47.74** | **+3.88** | 0.0007 |
| + SEP → Nemotron | 47.14 | +3.28 | 0.0025 |
| + SEP+ColQwen2 → Nemotron | 47.58 | +3.72 | 0.0010 |
| + Nemotron → SEP+ColQwen2 | 47.59 | −0.14 vs Nemotron | 0.89 n.s. |

**Rút ra:**
1. **Phát hiện "trùng lặp" ở §9 lặp lại trên một reranker thứ hai, hoàn toàn
   độc lập.** Đưa SEP / SEP+ColQwen2 vào trước Nemotron → *tệ hơn* một chút
   (47.1–47.6 vs 47.7). Chồng sau Nemotron → n.s. Hai cross-encoder khác nhau,
   cùng kết luận: **reranker được huấn luyện hấp thụ hết prior cấu trúc + arm
   visual.** Đây giờ là kết quả vững, không phải chuyện của riêng Voyage.
2. **Nhưng Nemotron một mình (47.74) chưa vượt stack miễn phí SEP+ColQwen2
   (48.35).** Trên physics/Pháp/text nó không phải lựa chọn mạnh nhất — nhưng nó
   là lựa chọn **miễn phí, không trần** mạnh nhất, và là reranker duy nhất đo
   trên đúng pool production.
3. **Chưa so 1-1 với Voyage** (Voyage đo trên `vidore_page`, chưa đo trên KDL).
4. **Chế độ đa phương thức (thêm ảnh trang): chạy được nhưng chậm ~15× (10.8s/câu
   → ~1h) và điểm relevance nhìn yếu (0.027 vs 0.9 ở text-mode)** — model card
   hứa +6–7% recall trên tài liệu *visual* (biểu đồ/bảng), mà trang vật lý chủ
   yếu là văn xuôi. Ghi nhận là chưa test; nếu theo, chạy subset nhỏ trước.

---

## 11. Bức tranh tổng thể — mạch xuyên suốt

1. **Corpus này topical, nhỏ, gold phân tán và dư thừa.** Không phải
   needle-in-haystack. Các phương pháp thiết kế cho needle-in-haystack (khu trú)
   đều thua ở đây.
2. **Recall đã xong. Ranking là vấn đề. Ranking trong cùng một file là phần khó.**
3. **Bi-encoder geometry không phân biệt được trang trong file** (DCW, sub-page
   MaxSim đều fail). Chỉ **cross-encoder được huấn luyện** làm được.
4. **Prior cấu trúc rẻ (SEP) và tín hiệu visual (ColQwen2) giúp — nhưng chỉ tới
   khi có một cross-encoder mạnh trong vòng lặp**, sau đó chúng thừa. Đã xác nhận
   trên **hai** reranker độc lập (Voyage §9, Nemotron §10).
5. Vậy kiến trúc đúng là: **hybrid retrieval tốt → một cross-encoder rerank mạnh
   (lý tưởng: miễn phí/self-host, lý tưởng: đa phương thức).** Mọi thứ khác chỉ
   là cộng thêm nhỏ.
   - Hiện tại: nếu chấp nhận API → Voyage 49.23. Nếu muốn miễn phí + không trần →
     Nemotron VL 47.74, hoặc stack SEP+ColQwen2 48.35 (không cần cả reranker).
     Chưa cái nào chạm SOTA physics 50.84.
6. **Generator quan trọng hơn retrieval ~20× cho QA.** Retrieval là bài toán con
   đáng làm cho đúng, nhưng đừng nhầm nó với đòn bẩy QA lớn nhất.

## 12. Hướng phát triển tiếp (chi tiết ở `retrieval_research_plan.md`)

| # | Việc | Vì sao |
|---|---|---|
| P1 | Visual arm ColQwen2-2B → Nemotron ColEmbed V2 4B / ColQwen3-4B | Dư địa lớn nhất; paper Snappy + leaderboard đều chỉ vào đây |
| P2 | Dense leg `text-embedding-3-small` → Qwen3-Embedding | Leg dense (41.1 một mình) là mắt xích yếu nhất |
| P3 | Reranker mạnh hơn / miễn phí: **Nemotron Rerank VL** (đang test), Qwen3-Reranker | §6: cross-encoder mạnh là đòn bẩy #1; thoát trần Voyage |
| P4 | α thích ứng: cross-encoder làm router (§5), hoặc LambdaMART/Metarank làm tầng fusion học được | Con đường duy nhất tới +8.60 headroom |
| P5 | Nemotron Rerank VL **kèm ảnh trang** | Gộp text + visual + cross-encoder vào 1 lời gọi |

**Mốc:** SOTA ViDoRe V3 physics = 50.84 (nemotron-colembed-8b, trung bình 6 ngôn
ngữ). Stack miễn phí hiện tại 48.35; Voyage 49.23. Khoảng cách tới SOTA ~1.5–2
điểm — và bài học §9 nói cách đóng nó là **một component mạnh hơn**, không phải
thêm lever.
