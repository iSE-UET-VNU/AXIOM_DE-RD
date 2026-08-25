# SEP — Structural Evidence Propagation

Tài liệu này giải thích chi tiết ý tưởng SEP và chạy thử từng bước trên **một
output KDL thật** của ViDoRe V3 physics.

---

## 1. Vấn đề mà SEP nhắm tới

Đo trên pool α=0.7 của physics (302 câu hỏi), lỗi retrieval chia làm ba dạng:

| dạng | tỉ lệ | mô tả |
|---|---|---|
| A | 7.0% | file chứa đáp án **không** nằm trong top-10 |
| B | **68.9%** | file đúng **đã** ở trong top-10, nhưng **sai trang** trong file đó |
| C | 24.2% | tất cả gold đã ở top-10 — không cần sửa |

Recall không phải vấn đề: `any-gold@100 = 98.3%`, `R@100 = 83.5`. Gần như câu
nào cũng đã có trang đúng trong top-100. **Vấn đề là thứ tự.**

Và con số quyết định — trong các ứng viên xếp hạng 11..100:

| nhóm ứng viên | n | tỉ lệ là gold |
|---|---|---|
| cùng file với một trang đã ở top-10 | 11,246 | **7.83%** |
| khác file | 15,934 | **0.46%** |

Chênh **17 lần**. Nói cách khác: một trang đến từ file mà hệ thống đã "tin", có
xác suất đúng cao gấp 17 lần một trang từ file lạ. Nhưng bộ xếp hạng hiện tại
chấm điểm **mỗi trang độc lập** và hoàn toàn không biết trang đó thuộc file nào.

SEP dùng chính thông tin đó.

---

## 2. Cây cấu trúc có sẵn miễn phí

Mỗi trang trong pipeline có một `unit_id` dạng:

```
physics::Cours_avance_Ch-0-Presentation_Cours_Avance#page=1
└─sub─┘  └──────────── file ────────────────────────┘ └page┘
```

Tức là **cây `file → page` đã nằm sẵn trong khoá**, không cần parser, không cần
LLM tóm tắt, không cần đổi index. SEP chỉ đọc `unit_id` và điểm số — **không hề
đọc text**, nên nó độc lập với bộ parse (chạy được trên KDL, chandra2, ViDoRe
text, light preparation).

> Xuất phát từ DISRetrieval (arXiv 2506.06313): một node cha có liên quan sẽ
> "kéo" các node lá thuộc cây con của nó lên. Bài báo dùng cây diễn ngôn RST.
> Nhưng chính ablation của họ cho thấy phần đắt tiền là phần bỏ được: RQ3 —
> truy hồi trên bản tóm tắt **kém hơn** truy hồi trên lá; RQ4 — đổi LLM tóm tắt
> chỉ thay đổi <0.5%. Nên ta giữ cơ chế tổng hợp, bỏ RST và bỏ tóm tắt LLM.

---

## 3. Công thức

Với mỗi trang `c` trong pool (top-100 của fusion α=0.7):

```
s'(c) = λ·s(c) + (1−λ)·[ β·A_file(c) + (1−β)·N(c) ]
```

| ký hiệu | ý nghĩa | giá trị dùng |
|---|---|---|
| `s(c)` | điểm gốc của chính trang đó (đã min-max về [0,1]) | — |
| `A_file(c)` | **trung bình top-m** điểm của các trang cùng file có trong pool | m = 3 |
| `N(c)` | bằng chứng từ **trang lân cận** cùng file, suy giảm theo khoảng cách | w = 2, γ = 0.5 |
| `λ` | trọng số giữa chính nó và cấu trúc | 0.5 |
| `β` | trọng số giữa "cả file" và "trang lân cận" | 0.75 |

Trong đó:

```
N(c)      = Σ  γ^|d| · s(page_c + d)      với 0 < |d| ≤ w, cùng file
A_file(c) = trung bình của m điểm cao nhất trong file của c
```

Cả `A_file` và `N` được min-max chuẩn hoá lại theo từng truy vấn trước khi cộng.

**Ba lựa chọn thiết kế quan trọng:**

1. **Trung bình, không phải tổng.** Tổng sẽ thiên vị file dài (file nhiều trang
   trong pool tự động thắng). Trung bình top-3 đo "chất lượng" của file, không đo
   "độ dài".
2. **β = 0.75 nghiêng về file.** Vì đo được: cùng-file cho lift 16.9×, còn
   lân cận chỉ thêm 1.63× *nữa* sau khi đã cùng file. Tín hiệu chính là file.
3. **Chỉ sắp xếp lại pool**, không thêm ứng viên mới. Do đó recall@100 **không
   đổi theo định nghĩa**, và mọi thay đổi đo được đều thuần tuý là thay đổi thứ tự.

---

## 4. Ví dụ chạy thật trên output KDL

Dữ liệu: `data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl`,
run `0183f2c78ac7802b`, 42 tài liệu / 1,674 trang.

### 4.1. Output KDL thô của một trang

Đây là các block thật mà KDL sinh ra cho trang thắng cuộc:

```json
{
  "component_id": "/page/1/Text/0",   "type": "Text",   "page": 1,
  "text": "Les grands domaines de la physique"
}
{
  "component_id": "/page/1/Figure/2", "type": "Figure", "page": 1,
  "text": "![Vitesse << c (3 10⁸ m/s) / Proche de c / Mécanique Classique / Relativité ..."
}
{
  "component_id": "/page/1/Text/3",   "type": "Text",   "page": 1,
  "text": "Un peu de Physique ... avancée"
}
```

Các block này được ghép thành text của trang, rồi gán khoá:

```
physics::Cours_avance_Ch-0-Presentation_Cours_Avance#page=1
```

SEP **chỉ dùng khoá này** (file + số trang), không dùng nội dung block.

### 4.2. Truy vấn

```
qid   : physics::267
query : "Décrivez le rôle des échelles de taille et de vitesse dans la
         structuration des domaines de la physique moderne."
gold  : 1 trang
```

### 4.3. Bảng xếp hạng TRƯỚC khi có SEP (α=0.7)

| hạng | trang | file | s(c) | gold? |
|---|---|---|---|---|
| 1 | `Cours_avance_ch-2-b-les-objets-quantiques-1#page=1` | Cours_avance_ch-2-b | 1.0000 | ✗ |
| **2** | **`Cours_avance_Ch-0-Presentation_Cours_Avance#page=1`** | **Cours_avance** | **0.9247** | **✓** |
| 3 | `Autrement_Ch-1b...#page=55` | Autrement | 0.7193 | ✗ |
| 4 | `Autrement_Ch-1b...#page=59` | Autrement | 0.6674 | ✗ |

→ **NDCG@10 = 63.09**. Trang đúng bị một trang khác chen lên trên.

### 4.4. Tính các thành phần cấu trúc

**Ứng viên hạng 1 (không phải gold).** File của nó có 4 trang trong pool, nhưng
điểm của chúng đều thấp:

```
s          = 1.0000        (cao nhất pool)
A_file_raw = 0.3708   →  A_file_norm = 0.3986      ← file yếu
N_raw      = 0.0458   →  N_norm      = 0.0876
```

**Ứng viên hạng 2 (gold).** File của nó chỉ có đúng 1 trang trong pool, và trang
đó chính là nó với điểm rất cao:

```
s          = 0.9247
A_file_raw = 0.9247   →  A_file_norm = 1.0000      ← file mạnh nhất pool
N_raw      = 0        →  N_norm      = 0.0000      (không có trang lân cận trong pool)
```

### 4.5. Áp công thức

```
Hạng 1:  s' = 0.5·1.0000 + 0.5·(0.75·0.3986 + 0.25·0.0876) = 0.6604
Hạng 2:  s' = 0.5·0.9247 + 0.5·(0.75·1.0000 + 0.25·0.0000) = 0.8373   ←
```

### 4.6. Bảng xếp hạng SAU khi có SEP

| hạng mới | (hạng cũ) | trang | s' | gold? |
|---|---|---|---|---|
| **1** | (2) | **`Cours_avance_Ch-0-Presentation...#page=1`** | **0.8373** | **✓** |
| 2 | (1) | `Cours_avance_ch-2-b...#page=1` | 0.6604 | ✗ |
| 3 | (3) | `Autrement...#page=55` | 0.6280 | ✗ |

→ **NDCG@10 = 100.0** (tăng **+36.91**).

### 4.7. Đọc ví dụ này thế nào

Trang đúng thắng **không phải vì điểm gốc của nó cao hơn** — điểm gốc của nó
thấp hơn (0.9247 < 1.0000). Nó thắng vì **bằng chứng ở cấp file** ủng hộ nó:
file của nó là file mạnh nhất trong pool, còn đối thủ tuy có 1 trang điểm cao
nhưng 3 trang còn lại cùng file đều yếu — dấu hiệu của một cú khớp lẻ loi,
có thể do trùng từ khoá ngẫu nhiên.

Đây đúng là ý nghĩa của `A_file` dùng **trung bình top-m**: nó phạt file mà chỉ
có một trang "may mắn" trúng, và thưởng file có bằng chứng nhất quán.

Chạy lại ví dụ này: `python research/experiments/vidore_sep_trace.py`

---

## 5. Kết quả đo được

Trên cấu hình production (KDL + chunk 512/128 + MaxP + α=0.7), 302 câu hỏi,
kiểm định hoán vị bắt cặp 10,000 lần:

| | NDCG@10 | R@10 | Correct_only | Correct+partial |
|---|---|---|---|---|
| baseline (bản tái lập) | 43.86 | 46.73 | 50.17 | 91.03 |
| **+ SEP (λ=0.5)** | **46.27** | **48.88** | 51.83 | 89.70 |
| Δ | **+2.41** (p=0.0004) | +2.15 | +1.33 (n.s.) | −1.33 (n.s.) |

Ổn định giữa hai nửa dữ liệu: fold0 = 46.22, fold1 = 46.33 (lệch 0.11).

**Chi phí:** +0.3 ms/truy vấn (18.5 → 18.8 ms). Không gọi API, không GPU, không
đổi index, không thêm gì ở giai đoạn offline.

**SEP cũng chuyển được sang parse khác** vì nó không đọc text:

| arm | baseline | + SEP | Δ | p |
|---|---|---|---|---|
| KDL physics | 43.03 | 44.75 | +1.73 | 0.0119 |
| KDL pharmaceuticals | 56.36 | 57.77 | +1.41 | 0.0021 |
| light preparation (pdf-inspector) | 43.09 | 44.62 | +1.53 | 0.0342 |

---

## 6. Giới hạn — cần nói rõ khi trích dẫn

**a) SEP chỉ chạy được trên corpus có tài liệu ngắn.** Nó cần bằng chứng cấp
file đủ sức thu hẹp "trang nào đúng". Có một phép sàng lọc **miễn phí** báo
trước điều đó (chỉ cần corpus + qrels, không cần embedding):

| subset | trang/file | gold chiếm % file của nó | kết quả |
|---|---|---|---|
| physics | 39.9 | 14.6% | **+2.02** |
| pharmaceuticals | 44.5 | 12.0% | **+2.17** |
| hr | 79.3 | 9.5% | +0.54 n.s. |
| industrial | 194.2 | 5.2% | n.s. |
| computer_science | 680.0 | 0.8% | n.s. |

Dự đoán được đăng ký **trước khi chạy** và đúng 5/5. Ngưỡng khoảng **gold ≥ 10%
số trang của file**. Với industrial (194 trang/file), `A_file` bị pha loãng và
SEP mất tác dụng.

**b) SEP không phải ý tưởng mới.** Đây là cluster hypothesis / score
regularization (Diaz 2005). Đóng góp thật nằm ở phần chẩn đoán và ở phép sàng
lọc điều kiện áp dụng, không phải ở công thức.

**c) β = 0.75 được chọn trên physics**, nên p-value của physics là lạc quan.
Khoảng trung thực là **+1 đến +2**. Điểm sạch nhất là pharmaceuticals (+1.41,
cấu hình giữ nguyên, chưa từng nhìn thấy subset đó).

**d) SEP không cải thiện QA.** Cả hai cột QA đều không có ý nghĩa thống kê và
ngược chiều nhau. Nguyên nhân đã đo: gold dư thừa (trung bình 7.21 trang gold,
chỉ cần ~2 trang để trả lời), nên trần QA của retrieval chỉ khoảng 3.6pp.
**Trích dẫn SEP như một kết quả retrieval, không phải kết quả end-to-end.**

Để so sánh, giữ nguyên retrieval và chỉ đổi generator (n=120):

| generator | Correct_only |
|---|---|
| deepseek-v4-flash | 44.54 |
| gpt-5.2 | **71.67** (+27.73, p=0.0001) |

Tức là đòn bẩy cho điểm end-to-end nằm ở generator, không nằm ở retrieval.

---

## 7. Đã khai thác hết long-distance evidence chưa? — Rồi, và mở rộng thêm thì hỏng

**Nới cửa sổ lân cận `w`** (bằng chứng xa hơn về mặt trang), so với SEP hiện tại
(w=2, γ=0.5), kiểm định bắt cặp:

| w | γ=0.5 | γ=0.8 |
|---|---|---|
| 2 (hiện tại) | 46.27 | 45.94 (−0.33, p=0.012) |
| 3 | 46.21 (−0.05 n.s.) | 45.93 (−0.34 n.s.) |
| 5 | 46.17 (−0.10 n.s.) | 45.63 (−0.64, p=0.033) |
| 8 | 46.19 (−0.08 n.s.) | 45.60 (−0.67 n.s.) |
| 12 | 46.19 (−0.08 n.s.) | 45.37 (−0.89, p=0.017) |
| 20 | 46.19 (−0.08 n.s.) | 45.39 (−0.88, p=0.019) |

**Không có gì để lấy thêm.** Lý do: `A_file` vốn đã tổng hợp **toàn bộ file**,
tức khoảng cách không giới hạn. Số hạng lân cận `N` chỉ bổ sung tinh chỉnh cục
bộ ở ±2; nới rộng nó chỉ lặp lại thông tin `A_file` đã có.

**Lan truyền nhiều bước** (kiểu khuếch tán trên đồ thị) thì **có hại rõ rệt**:

| số bước | NDCG@10 | so với 1 bước |
|---|---|---|
| 1 (hiện tại) | 46.27 | — |
| 2 | 44.27 | **−2.00** (p=0.0007) |
| 3 | 39.63 | **−6.64** (p=0.0002) |

Mỗi vòng làm phân bố điểm mượt thêm; đến vòng thứ hai thì cấu trúc file đã lấn
át tín hiệu gốc của từng trang.

### Lỗ hổng thật: bằng chứng **liên file**

| nhóm | tỉ lệ | baseline | +SEP | Δ |
|---|---|---|---|---|
| gold trải trên >1 file | 13.6% (41/302) | 42.38 | 43.78 | **+1.40** |
| gold trong 1 file | 86.4% | 44.09 | 46.66 | **+2.57** |

SEP giúp nhóm liên file **kém gần một nửa**. Đúng theo thiết kế: nó tổng hợp
*trong* một file, nên khi bằng chứng nằm rải ở nhiều tài liệu thì nó không có gì
để cộng dồn. Đây là hướng còn bỏ ngỏ, nhưng chỉ tác động lên 13.6% câu hỏi.

---

## 8. Đã khai thác hết visual chưa? — Có dùng, nhưng gần như không có tác dụng

Ta **không** có ảnh trang. Cái ta có là bản KDL diễn giải hình ảnh thành text:
1,648 block `Figure` có nội dung, cộng 1,043 `Caption` và 425 `Table`.

Về khối lượng thì phần "visual" này rất lớn:

| loại | % tổng số ký tự |
|---|---|
| **Figure** | **29.6%** |
| **Table** | **26.2%** |
| còn lại (Text, SectionHeader, Equation) | 44.1% |

Nhưng khi bỏ chúng ra khỏi text của trang (đo bằng BM25, không cần API):

| arm | chars/trang | BM25 NDCG@10 | Δ | p |
|---|---|---|---|---|
| đầy đủ (hiện tại) | 1,330 | 37.46 | — | |
| bỏ Figure | 938 | 36.25 | −1.20 | 0.108 n.s. |
| bỏ Figure + Caption | 919 | 36.09 | −1.37 | 0.083 n.s. |
| bỏ Table | 984 | 36.97 | −0.48 | 0.511 n.s. |
| chỉ Text + SectionHeader | 507 | 35.62 | −1.83 | 0.105 n.s. |

**Bỏ 62% số ký tự chỉ mất 1.83 điểm, và không có ý nghĩa thống kê.** Nội dung
suy ra từ hình ảnh chiếm gần 30% khối lượng nhưng đóng góp rất ít cho việc xếp
hạng. Khớp với kết quả cũ trên chandra2: image descriptions +0.99 (p=0.25).

Nghĩa là **visual đã được dùng — dưới dạng text — và dạng đó gần như vô dụng cho
retrieval.** Câu hỏi còn mở là truy hồi bằng **ảnh thật** (visual encoder) có
khác không. Ba rào cản, đều là rào cản thật:

1. Ảnh trang **chưa tải về** (442 MB – 2.2 GB mỗi subset).
2. GPU chỉ chạy được qua Google Colab.
3. Hạn mức API đã hết.

Tham chiếu để định lượng cơ hội: ColEmbed đạt 91.0 (ViDoRe V1) / 63.5 (V2) bằng
late interaction trên ảnh; các visual retriever chạy 43.2–48.5 trên physics. Và
Bảng 6 của chính bài đó cho thấy **bi-encoder ảnh một vector + reranker** đạt
0.9064 so với 0.9106 của late interaction đầy đủ, ở mức **3.8 GB thay vì 10,311
GB cho mỗi triệu trang** — tức bản rẻ không cần đổi index contract.

**Kết luận ngắn:** long-distance đã hết dư địa; visual-dưới-dạng-text đã dùng và
không đáng kể; visual-dưới-dạng-ảnh là dư địa thật nhưng đang bị chặn bởi hạ tầng
chứ không phải bởi ý tưởng.

Chạy lại:
```bash
python research/experiments/vidore_longrange.py
python research/experiments/vidore_visual_ablation.py
```
