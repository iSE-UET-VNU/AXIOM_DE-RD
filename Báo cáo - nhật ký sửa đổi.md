# Nhật ký sửa đổi báo cáo thực tập

Bản gốc: `Trần Minh Nam (23021646) - Báo cáo thực tập -1.pdf` (23 trang, Google Docs)
Bản sửa: `Trần Minh Nam (23021646) - Báo cáo thực tập.docx`

## Bước 0 — Dựng lại file .docx (2026-09-26)

Nội dung giữ nguyên từng chữ. Chỉ thay đổi hình thức.

### Toàn bài
| Hạng mục | Bản gốc | Bản sửa |
|---|---|---|
| Font | Lẫn Times New Roman, Cambria, Arial | Times New Roman toàn bài (riêng dòng lệnh ở mục IV dùng Courier New) |
| Cỡ chữ | Không đồng nhất | Thân bài 13; bảng và chú thích 12; tiêu đề cấp 1 là 14 đậm, cấp 2 là 13 đậm, cấp 3 là 13 đậm nghiêng |
| Lề | Không đều | Trái 3 cm, phải 2 cm, trên 2 cm, dưới 2 cm |
| Giãn dòng, căn lề | Không đều | Giãn dòng 1,3, căn đều hai bên, thụt đầu dòng 1 cm ở đoạn văn thân bài |
| Ký hiệu gạch đầu dòng | ● và ○ | • (cấp 1) và – (cấp 2) |
| Số trang | 23 | 27 (do lề và giãn dòng mới) |

### Trang bìa
- Thêm khung viền đôi.
- Tên trường in đậm, có đường kẻ ngang bên dưới; logo to hơn.
- "ĐỀ TÀI:" tách thành một dòng riêng, tên đề tài nằm trọn trên dòng tiếp theo.
- Giảng viên hướng dẫn, sinh viên, mã sinh viên, lớp được xếp thành 2 cột thẳng hàng. Bản gốc lệch nhau.
- Bỏ số trang ở bìa. Trang Lời cảm ơn vẫn đánh số 2 như bản gốc.

### Mục lục
- Có dấu chấm dẫn, số trang được cập nhật theo bố cục mới.
- Bổ sung các mục có trong thân bài nhưng bản gốc thiếu trong mục lục: III.c.3, III.c.4, IV.a, IV.b, IV.c, V.a.
- Sửa lề thụt sai của mục "1. Thực nghiệm Module chunking và embedding".

### Bảng, hình, công thức
- Tất cả các bảng được dựng lại thành bảng Word thật, hàng tiêu đề tô xám. Chú thích bảng đặt phía trên, in đậm nghiêng.
- Bảng 3: sửa tiêu đề cột bị vỡ dòng ("NDCG @10" → "NDCG@10", "R@1 0" → "R@10").
- Bảng 3: in đậm hàng "Kết hợp SEP và ColVec-8b". Đây là thay đổi mình tự thêm, bạn bỏ đi được.
- Bảng 8: in đậm hàng "Visual theo yêu cầu, cổng Kf = 3" cho khớp chú thích "chữ đậm là cấu hình đề xuất".
- Bảng 7 và Bảng 8 không còn bị thụt vào trong danh sách.
- Hình 1, Hình 2: lấy lại ảnh gốc, chú thích in đậm nghiêng.
- Công thức: căn giữa, in nghiêng, giữ nguyên nội dung. Sửa lỗi hiển thị "I DCG@10" thành "IDCG@10".
- Hai đoạn kết luận đóng khung (sau Bảng 3 và sau Bảng 4): khung xám, chữ nghiêng.

### Mục IV
- Các dòng lệnh đặt trong khối nền xám và ngắt dòng bằng dấu "\" để không bị tràn.
- "pip install -e .." → "pip install -e ." (bản gốc bị dính dấu chấm cuối câu vào lệnh).

### Trang chữ ký và trang "Ý kiến đánh giá"
- Khối chữ ký ở Lời cảm ơn và Lời cam đoan được căn thẳng hàng.
- Trang "Ý kiến đánh giá": các dòng chấm được kéo đều hết chiều ngang. Dòng ngày ký đổi thành "Hà Nội, ngày …… tháng …… năm 20……".

## Bước 1 — Viết lại mục I và II (2026-09-28)

Theo nhận xét của cô: mô tả mục tiêu tổng thể và từng module trước, sau đó mới đến phân công. Đồng thời bỏ các chỗ trùng thông tin và bỏ lối viết nhãn kèm dấu ":", dấu ";", câu dài.

### Cấu trúc
| Bản gốc | Bản sửa |
|---|---|
| I.b Bối cảnh và động lực của bài toán | I.b Giới thiệu qua bài toán (theo mẫu của trường) |
| II.a Phân chia công việc trong nhóm | II.a Mô tả bài toán tổng thể (mới) |
| II.b Phát biểu bài toán truy xuất | II.b Các module và phần việc của em (mới), gồm 7 mục nhỏ |
| II.c Phạm vi và giả định | II.c Phân chia công việc trong nhóm |

### Nội dung
| Vị trí | Bản gốc | Bản sửa |
|---|---|---|
| I.a đoạn 1 | Câu dài | Tách thành câu ngắn, giữ nguyên ý |
| I.a đoạn 2 | Liệt kê 2 nhiệm vụ rồi liệt kê lại 2 module | Gộp thành một đoạn |
| I.a đoạn 3 | Kể trình tự các hướng đã thử, gồm cả reranking | Nêu ba kết quả chính: chunking, SEP kết hợp visual, thu hẹp phạm vi mã hoá |
| I.b | Nêu khó khăn kèm giải pháp (Uniform IR, offline, light preparation) | Chỉ nêu hai khó khăn, giải pháp chuyển sang II.a |
| II.a | (không có) | Đầu vào, đầu ra và bốn yêu cầu chung: truy vết, độc lập định dạng, thời gian phản hồi, tương thích môi trường |
| II.b.1–5 | 3 gạch đầu dòng mô tả các mảng | Bảng 1 (đầu vào, đầu ra) và một đoạn mục tiêu, yêu cầu cho mỗi module |
| II.b.6 | Phát biểu bài toán truy xuất | Giữ nguyên ý, tách câu ngắn |
| II.b.7 | Phạm vi, kèm công thức NDCG và Recall | Ba giả định, công thức chuyển sang III.a.5 |
| II.c | Bảng phân công và 3 gạch đầu dòng | Bảng 2, chỉ giữ đoạn về phần việc của em và bạn Tân |
| III.a.5 | Không có công thức | Nhận công thức NDCG@10, DCG@10, Recall@10 từ II.c cũ (giữ nguyên chữ) |
| IV.c | Nhắc lại PyTorch 2.2.2 và ONNX | Dẫn chiếu về mục II.a |

### Đánh số và dẫn chiếu
| Vị trí | Bản gốc | Bản sửa |
|---|---|---|
| Chú thích hình, bảng | "Hình 1:", "Bảng 1:" | "Hình 1.", "Bảng 1." |
| Hình 1 chú thích | "Đường ống xử lý dữ liệu…" | "Quy trình xử lý dữ liệu…" (chữ trong ảnh chưa đổi) |
| Bảng phân công | Bảng 1 | Bảng 2 |
| Bảng chunking | Bảng 2 | Bảng 3 |
| Bảng kết quả retrieval | Bảng 3 | Bảng 4 |
| Bảng SEP theo tập con | Bảng 4 | Bảng 5 |
| Bảng end-to-end | Bảng 5 | Bảng 6 (lấp chỗ thiếu Bảng 6) |
| III.c.4 | "Cấu hình tốt nhất ở Bảng 5" | "ở Bảng 4" (bản gốc dẫn nhầm) |
| Lời cam đoan | "mục II.c" | Giữ nguyên, vì phân công giờ đúng là II.c |
| Mục lục | II.a–c | II.a–c, thêm II.b.1–7 |

## Bước 2 — DISRetrieval, SEP và mục so sánh (2026-09-28)

Theo nhận xét của cô: phân tích DISRetrieval là gì, điểm mạnh, điểm yếu, SEP đóng góp gì và khác gì so với DISRetrieval và các giải pháp khác. Mọi khái niệm được giải thích ở lần đầu nhắc tới.

### Cấu trúc mục III
| Bản gốc | Bản sửa |
|---|---|
| III.a.1–3 | Giữ nguyên |
| (không có) | III.a.4 Truy xuất theo cấu trúc tài liệu và DISRetrieval (mới) |
| III.a.4 Visual document retrieval và reranking | III.a.5 |
| III.a.5 Dữ liệu và thước đo đánh giá | III.a.6 |
| III.b Giải pháp đề xuất | III.b Giải pháp đề xuất và kết quả thực nghiệm |
| III.c.1–4 Kết quả thực nghiệm | III.b.6–9 (nội dung chưa đổi) |
| III.e Liên hệ và so sánh | III.c (theo mẫu của trường) |

### Nội dung
| Vị trí | Bản gốc | Bản sửa |
|---|---|---|
| III.a.4 | (không có) | Nút lá và nút trong, RAPTOR, RST và discourse tree có ví dụ, ba bước của DISRetrieval, bốn baseline (flatten-chunk, flatten-sentence, RAPTOR, Bisection), bốn bộ dữ liệu, kết quả, bốn điểm yếu |
| III.b.3 | "Ý tưởng xuất cơ chế… cới mỗi trang…", không có số liệu chẩn đoán | Chẩn đoán lỗi (7,0% / 68,9% / 24,2%, 98,3%, 7,83% so với 0,46%, 1,63 lần), SEP mượn gì từ DISRetrieval, cây tài liệu và trang, công thức, ba lựa chọn thiết kế, chi phí |
| III.b.3 hạn chế | Một đoạn giải thích A_file bị pha loãng | Một câu dẫn tới III.b.7 |
| III.c | 3 đoạn, có câu "giảm hàng trăm lần chi phí" (không có số liệu) | Bảng 9 và bốn đoạn so sánh với DISRetrieval, RAPTOR, visual retrieval; nêu rõ SEP +2,08 và +1,42 điểm NDCG@10 |
| Tài liệu tham khảo [2] | H. Zhang et al., "DISRetrieval: Harnessing Discourse Structure for Long Document Retrieval" | H. Chen, Y. Yang, Y. Li, M. Zhang, B. Hu, M. Zhang, "Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering" |

### Giải thích thuật ngữ thêm vào mục I, II
| Vị trí | Thuật ngữ | Giải thích thêm |
|---|---|---|
| I.a | parsing | (bóc tách nội dung tài liệu thành văn bản có cấu trúc) |
| II.a | PyTorch, ONNX Runtime | (hai thư viện chạy mô hình học sâu) |
| II.b.1 | pdf-inspector, BM25, Kf | Mỗi thuật ngữ có một giải thích ngắn |
| II.b.2 | KDL | (frontier parser nano, bộ công cụ parsing tài liệu thành dữ liệu có cấu trúc) |
| II.b.3 | MaxP, plugin, SHA-256 | Mỗi thuật ngữ có một giải thích ngắn |
| II.b.4 | top-k | (k trang có điểm cao nhất) |
| II.b.7 | NDCG@10, Recall@10 | Mỗi thước đo thêm một câu về ý nghĩa |

### Dẫn chiếu
| Vị trí | Bản gốc | Bản sửa |
|---|---|---|
| II.b.4 | mục III.c.2 | mục III.b.7 |
| II.b.7 | mục III.a.5 | mục III.a.6 |
| III.b.7 (cũ III.c.2) | "Mục III.b.6 đã lập luận" | "Mục III.b.3 đã lập luận" |
| III.c (cũ III.e) | mục III.c.3 | mục III.b.9 |
| Bảng mới | (không có) | Bảng 9 |

## Bước 3 — Mục V và thời gian chạy (2026-09-28)

Theo nhận xét của cô: phân tích kết quả công việc đạt được, và phân tích hạn chế của giải pháp trước khi đề xuất hướng phát triển.

### Cấu trúc mục V
| Bản gốc | Bản sửa |
|---|---|
| V.a Kỹ năng và kiến thức thu thập được | V.a Kết quả đạt được (mới) |
| V.b Hướng phát triển tiếp theo | V.b Kỹ năng và kiến thức thu thập được |
| (không có) | V.c Hạn chế của giải pháp hiện tại (mới) |
| (không có) | V.d Hướng phát triển tiếp theo |

### Nội dung
| Vị trí | Bản gốc | Bản sửa |
|---|---|---|
| V.a | (không có) | Bảng 10 tóm tắt kết quả (chunking, chất lượng retrieval, SEP, thời gian chạy) và một đoạn phân tích |
| V.b | 3 nhãn đậm kèm ":", danh sách năm thói quen nghiên cứu, chi tiết plugin và SHA-256 | Hai đoạn ngắn về kiến thức và kỹ năng. Bỏ danh sách thói quen, bỏ chi tiết đã có ở II.b.3 |
| V.c | (không có) | Ba đoạn hạn chế: SEP với tài liệu dài, nhánh visual (giấy phép, độ trễ phần đuôi, số liệu mô phỏng), retrieval chưa cải thiện câu trả lời |
| V.d | 6 hướng dạng nhãn đậm kèm ":" | Ba đoạn, mỗi đoạn trả lời một hạn chế ở V.c |
| V.d | Hướng "mở rộng đánh giá trên DocBench" | Bỏ |
| V.d | Hướng "tinh gọn quy trình linh hoạt (adaptive)" | Bỏ |
| V.d | Câu "suy giảm NDCG@10 khi mở rộng kho đa miền" | Bỏ |
| Đoạn cuối mục V | "Đóng gói thành dịch vụ độc lập…" | Bỏ, vì đã có ở II.b.5 |

### Thời gian chạy
| Vị trí | Bản gốc | Bản sửa |
|---|---|---|
| II.b.6 | "T_first là thời gian từ lúc kho tài liệu được nạp đến lúc hệ thống trả lời được câu hỏi đầu tiên" | "T_first là thời gian người dùng phải chờ từ lúc đưa kho tài liệu vào hệ thống cho tới khi nhận được câu trả lời đầu tiên" |
| Bảng 8 | Cột: Thời gian tới câu trả lời đầu tiên, p50, p95, Truy vấn chậm nhất, Tổng 302 truy vấn | Cột: Chi phí offline, Thời gian tới câu trả lời đầu tiên, Online trung bình mỗi câu, p50, p95, Truy vấn chậm nhất. Bỏ cột tổng |
| Bảng 8, hai dòng visual theo yêu cầu | (không có) | Chi phí offline 1m 17s (nạp mô hình). Online trung bình 7.88 s (Kf = 3) và 8.18 s (Kf = 10), suy ra từ tổng thời gian cũ: (tổng − 77,2 s) / 302 |
| Bảng 8, dòng visual lập chỉ mục toàn kho | p95 ghi "-" | p95 ghi "—" cho thống nhất |
| III.b.9 | "thời gian tới câu trả lời đầu tiên và tổng thời gian 302 truy vấn gần như bằng nhau" | "gần như toàn bộ thời gian nằm ở chi phí offline" (vì đã bỏ cột tổng) |

## Bước 3b — Chuyển bảng thành văn để tiết kiệm chỗ (2026-09-28)

| Vị trí | Trước | Sau |
|---|---|---|
| II.b | Bảng 1 (đầu vào, đầu ra của 5 module) | Bỏ bảng. Mỗi đoạn module thêm một câu nêu đầu vào và đầu ra |
| II.b.1 | "Kết quả là Kf tài liệu ứng viên, với Kf là…" ở cuối đoạn | Chuyển lên câu đầu vào, đầu ra ở đầu đoạn |
| III.c | Bảng 9 (so sánh RAPTOR, DISRetrieval, SEP) và câu ghi chú | Bỏ bảng. Đoạn so sánh với DISRetrieval thêm hai câu về chi phí tiền xử lý, bỏ câu "Nhờ vậy, SEP không có chi phí tiền xử lý" bị trùng |
| V.a | Bảng 10 (tóm tắt kết quả) | Bốn gạch đầu dòng, mỗi dòng dẫn tới mục chi tiết |

### Đánh số lại bảng
| Cũ | Mới |
|---|---|
| Bảng 2 Phân công | Bảng 1 |
| Bảng 3 Chunking | Bảng 2 |
| Bảng 4 Kết quả retrieval | Bảng 3 |
| Bảng 5 SEP theo tập con | Bảng 4 |
| Bảng 6 End-to-end | Bảng 5 |
| Bảng 7 Phạm vi mã hoá | Bảng 6 |
| Bảng 8 Chi phí thời gian | Bảng 7 |

Mọi dẫn chiếu tới bảng trong bài đã được đổi theo.

## Bước 4a — Văn phong mục III.a.1–3 (2026-09-28)

| Vị trí | Trước | Sau |
|---|---|---|
| III.a.1 tiêu đề | "đơn vị truy hồi" | "đơn vị truy xuất" |
| III.a.1 | "điểm dung hòa tối ưu… phù hợp hoàn hảo… hoàn toàn tương thích…" | Bỏ từ phóng đại, nói thẳng lý do chọn trang |
| III.a.1 | ViDoRe V3, MMDocIR chưa giải thích | Thêm một câu giải thích |
| III.a.1 | Hai loại tổn thất viết nhãn đậm kèm ":" | Viết thành câu |
| III.a.2 tiêu đề | "Các chiến lược phân đoạn văn bản (Chunking)" | "Các chiến lược chunking" |
| III.a.2 | "Cắt cố định (Fixed-size chunking):…" | Giữ tên tiếng Anh, viết thành câu |
| III.a.2 | Giải thích RAPTOR lần hai | Nêu tên và dẫn tới III.a.4 |
| III.a.2 | "Điều này sẽ được thấy rõ hơn ở phần III.b.1" | "Mục III.b.1 so sánh trực tiếp…" |
| III.a.3 | "Để khai thác tối đa ưu điểm…", nhãn đậm kèm ":" | Câu mở đầu ngắn, gạch đầu dòng dạng câu dẫn kèm ":" |
| III.a.3 | text-embedding-3-small chưa giải thích | Thêm "(mô hình embedding của OpenAI)" |
| III.a.3 | "thang đo và hàm phân phối xác suất hoàn toàn khác biệt" | "thang đo rất khác nhau" |
| Công thức min-max, hybrid | "-" và "*" | "−" và "·" |

## Bước 4 — Rà soát toàn bài (2026-09-28)

Rà toàn bộ báo cáo theo ba nhóm nhận xét: các nhận xét của cô, các quy tắc văn phong của Nam (không dùng nhãn kèm ":", ";", gạch ngang, không dịch sát nghĩa, giải thích thuật ngữ ở lần đầu nhắc tới, không cắt câu cụt) và các chỗ trùng thông tin. Báo cáo giờ dài 31 trang.

### Theo mục
| Vị trí | Trước | Sau |
|---|---|---|
| Trang bìa | "PLATOFORM", "Nguyễn Thu Trang" | "PLATFORM", "TS. Nguyễn Thu Trang" |
| I.a | Câu cụt, "Module này… Module thứ hai…" | Nối câu. Giải thích chunking, embedding, parsing, mã hoá ảnh trang |
| I.b | "PDF scan" chưa giải thích | Thêm giải thích |
| II.a | ingestion, offline, online chưa giải thích | Thêm giải thích. t_q gọi là "thời gian trả lời mỗi câu hỏi" |
| II.b.1–5 | Câu cụt, module 4 và 5 lặp ý | Viết liền mạch, bỏ phần lặp |
| II.b.7 | "Chi phí… tổng thời gian chạy" | "chi phí offline, T_first và t_q" |
| Bảng 1 | "pipeline MVP on-request", "Light preparation" chưa giải thích | "quy trình MVP (phiên bản tối thiểu chạy được) xử lý theo yêu cầu", giải thích light preparation |
| III.a.1 | "Vì vậy, việc chọn đơn vị… rất quan trọng" (không nối ý) | Nêu lý do chọn đơn vị. Nối câu ở gạch đầu dòng về trang |
| III.a.2 | token, LLM chưa giải thích | Thêm giải thích |
| III.a.4 | Ví dụ RST và phần bốn bộ dữ liệu bị cắt câu cụt | Gộp thành câu liền mạch |
| III.a.5 | "Truy xuất tài liệu thị giác… tương tác muộn… bài toàn… nút chặt… tại liệu… chéo toàn diện", "Trong đó:" | Viết lại theo trình tự vấn đề, cách làm, tên kỹ thuật. Giữ 8 dòng ký hiệu MaxSim |
| III.a.6 | Nhãn "Thước đo chính:", câu dài | Giải thích NDCG@10 bằng lời trước công thức, viết thành đoạn văn |
| III.b.1 | Nhãn kèm ":", chỉ nói recall tăng, Qasper, UIT-ViQuAD, ISE Challenge chưa giải thích | Nói đủ cả ba bộ dữ liệu (ISE Challenge giảm 1,7%), giải thích từng bộ |
| Bảng 2 | "Fixed 512 - Overlap 128", "+10.8% tương đối" | "Fixed 512, overlap 128", "+10,8%" |
| III.b.2 | "pipeline cơ sở", "phân tích từ vựng" | "quy trình retrieval cơ sở", "tách từ". Giải thích pool |
| III.b.3 | Câu cụt ở phần chẩn đoán | Nối câu |
| III.b.4 | "ngiên", giải thích lại MaxSim trùng III.a.5 | Sửa lỗi, dẫn chiếu III.a.5. Giải thích OCR, ColVec-8b, ColQwen2-2B |
| III.b.5 | "Voyage rerank-2.5 [13]" ([13] là OmniDocBench), "chú ý chéo toàn diện" | Bỏ Voyage (không có trong bảng kết quả), giải thích mô hình Nemotron |
| III.b.6 | "chỉ thêm 0.3 mili giây" (bảng ghi +1,6 ms), "chi phí online bằng không", khung tóm tắt lặp | "+1,6 ms", "chi phí nằm gần như hoàn toàn ở phía offline", bỏ khung. Nêu rõ reranking không vào cấu hình cuối. Giải thích MTEB |
| Bảng 3 | "Pipeline cơ sở", "Hiệu năng tương đối (Relative Δ)" | "Quy trình cơ sở", "Thay đổi NDCG@10" |
| III.b.7 | "VidoreV3" thiếu dấu chấm, khung tóm tắt lặp | "ViDoRe V3.", bỏ khung, thêm câu "SEP chỉ nên bật cho kho tài liệu ngắn" |
| III.b.8 | "Phần nghiên này", câu dài | Sửa lỗi, tách câu. Giải thích end-to-end, DeepSeek-V4-Flash, gpt-5.2 |
| III.b.9 | "đỗ trễ", "( nhanh", "ở Kf", "Chi phí", "bộ nhớ đệm", "cổng", nhãn "Cold start:", đoạn kết "Với những tìm kiếm ở trên…" | Sửa hết. Nói rõ 1 giờ 58 phút gồm hai nhánh, còn Bảng 6 và 7 chỉ dùng nhánh visual. Giải thích cache, p50, p95, batch, batching, quantization. Bỏ đoạn kết |
| Bảng 6, Bảng 7 | "Colvec1.1-8B", "Pipeline gốc", "cổng Kf" | "ColVec-8b", "Quy trình gốc", "Kf" |
| IV | Nhãn kèm ":" trước mỗi lệnh, OpenRouter chưa giải thích | Câu dẫn trước mỗi lệnh, giải thích OpenRouter |
| V.a | Dòng thời gian chạy trộn với cấu hình 53,15 | Ghi rõ "cấu hình chỉ dùng nhánh visual và Kf = 3", NDCG@10 50,83 so với 51,63 |
| V.c | "hơn 600 trang" như thể trên physics | "Khi kho dữ liệu lớn hơn và gồm nhiều miền, …" |
| V.d | Giải thích batching, quantization lần hai | Bỏ giải thích lặp |

### Toàn bài
| Hạng mục | Trước | Sau |
|---|---|---|
| Dấu thập phân | Lẫn "." và "," | "," trong văn bản và bảng. Dấu "." chỉ dùng để phân cách hàng nghìn |
| Chính tả | Lẫn "hóa/hoá", "tỷ/tỉ", "khóa/khoá" | "hoá", "tỉ", "khoá" |
| Thuật ngữ | "truy hồi", "bộ nhớ đệm", "ngoại tuyến", "cổng" | "truy xuất", "cache", "offline", bỏ "cổng" |

### Còn tồn tại
- Bảng 2, dòng ISE Challenge: theo docs/incorpus_benchmark_findings.md (commit caa12be), dòng 104.927 chunk là chiến lược recursive_400 chứ không phải semantic, và fixed_512 có R@10 = 0,571 chứ không phải 60,21%. Chờ Nam xác nhận.

## Bước 5 — Mục I.b và Hình 1 (2026-09-28)

Đồng bộ các sửa của Nam trong v2.pdf vào bản dựng, rồi viết lại I.b theo đúng vai trò của mục này trong mẫu của trường. I.b chỉ giới thiệu bài toán, còn giải pháp và mô tả chi tiết nằm ở mục II và III.

| Mục | Trước | Sau |
|---|---|---|
| Lời cam đoan | "dưới sự hướng dẫn của TS.", "gồm cả những hướng tiếp cận đã bị bác bỏ" | Theo v2: "của cô TS.", bỏ vế "gồm cả những hướng tiếp cận đã bị bác bỏ" |
| I.a, đoạn 1 và 2 | Bản cũ | Theo v2 của Nam. Sửa lỗi gõ "chunking-mebeding" thành "chunking-embedding" và thêm "qua parsing" vào câu "tài liệu đã thành" |
| I.b | Hai khó khăn, kèm một đoạn nêu giải pháp (Uniform IR, offline, light preparation, refinement) | Nêu bài toán bằng lời (đầu vào, đầu ra, vì sao quan trọng) và hai thách thức "Sự đa dạng của dữ liệu đầu vào", "Giới hạn về thời gian phản hồi". Bỏ đoạn giải pháp vì lặp với II.a và có câu "tác vụ nặng chạy offline một lần duy nhất" mâu thuẫn với thách thức thứ hai |
| Tên mục II.b | "Các module và phần việc của em" | "Các module và phân công công việc" (theo v2, sửa lỗi gõ "phần công") |
| Hình 1 | Ảnh cũ có chữ "Đường ống", thiếu module Tích hợp AXIOM, nhiều chú thích đỏ và tên thành viên. Đặt ở I.b | Vẽ lại với hai giai đoạn (chuẩn bị nhẹ trên toàn bộ kho, xử lý sâu trên tập con), đủ năm module đánh số như II.b, đánh dấu phạm vi của em. Chuyển xuống đầu II.b |

## Bước 6 — Sắp xếp lại mục II theo v3 (2026-09-28)

Nguyên tắc là mỗi khái niệm chỉ định nghĩa một lần, ở chỗ đầu tiên cần đến nó. Đồng bộ các sửa của Nam trong v3.pdf vào bản dựng.

| Mục | Trước | Sau |
|---|---|---|
| II.a, Yêu cầu 3 | Nhắc T_first, t_q và trỏ tới "mục II.b.6" | Định nghĩa trực tiếp chi phí offline, T_first, t_q và nêu sự đánh đổi giữa chúng. Giữ câu kết theo v3 "chỉ cần xử lý các phép tính truy xuất nhẹ và sinh câu trả lời nhanh chóng" |
| II.a, đoạn cuối | Không có | Thêm đoạn nói bốn yêu cầu là tiêu chí đánh giá giải pháp (chuyển từ II.b.6 cũ) |
| II.b.3 | Có câu plugin và SHA-256 | Bỏ (theo v3) |
| II.b.4 | Có hai câu về t_q và bật tắt độc lập | Bỏ (theo v3), sửa dấu ". ." |
| II.b.6 Phát biểu bài toán retrieval | Mục riêng | Bỏ. Phần hàm điểm chuyển sang III.a.1, phần chi phí chuyển sang II.a |
| II.b.7 Phạm vi và giả định | Ba giả định | Bỏ. Đơn vị trang đã có ở III.a.1, thước đo đã có ở III.a.6, giả định parsing cố định chuyển sang III.a.6 |
| II.c Phân chia công việc | Mục II.c | Thành II.b.6 (theo v3). Bỏ "tích hợp tầng reranking" ở đoạn phân công |
| Lời cam đoan | "ghi rõ ở mục II.c" | "ghi rõ ở mục II.b.6" |
| III.a.1 | Không có phát biểu hình thức | Thêm hàm điểm s_page(q, c) (theo v3, sửa lại thành câu hoàn chỉnh). Gộp đoạn MaxP với đoạn tổn thất |
| III.a.4 | "Phương pháp xử lý từ các nút gần lá", "mô hình Llama-3.1-8B (một LLM mã nguồn mở 8 tỉ tham số) sẽ tóm tắt" | "Phương pháp DISRetrieval xử lý từ các nút lá", "phương pháp sẽ sử dụng mô hình Llama-3.1-8B để tóm tắt" (theo v3) |
| III.a.6 | "Chi phí được đo bằng ba đại lượng là…" | Trỏ về định nghĩa ở II.a, thêm đoạn thiết lập đánh giá (parsing chung, cùng 302 câu hỏi, cùng thước đo) |
| III.b.6 | "đo trên cùng kết quả parsing, cùng thước đo và cùng 302 câu hỏi" | "theo thiết lập đánh giá ở mục III.a.6" |
| Hình 1 | "Phạm vi công việc của em" | "Phạm vi công việc trong báo cáo" |

### Còn tồn tại
- Hình 2 vẫn là ảnh cũ, còn chữ "ngoại tuyến", "trực tuyến", "Nhánh thị giác".

## Bước 7 — Tái cấu trúc toàn bài theo mẫu và theo mạch bài báo (2026-09-28)

Lý do: báo cáo chia quá nhiều mục con (21 mục cấp ba), trộn giải pháp với kết quả, đặt dữ liệu và thước đo trong phần lý thuyết, và nhắc SEP trước khi giải thích. Khung mới giữ đúng các mục của mẫu, bên trong viết theo mạch bài báo (bối cảnh, phương pháp, thực nghiệm, thảo luận). Đồng bộ các sửa của Nam trong v4.pdf. Báo cáo giảm từ 31 xuống 27 trang, phần nội dung còn khoảng 20 trang.

| Mục | Trước | Sau |
|---|---|---|
| I.a | "điểm khác biệt lớn giữa AXIOM và các hệ thống hỏi đáp thông thường", "tìm kiếm theo nghĩa", "Chi tiết từng phần được trình bày ở mục III.b" | Theo v4: "điểm quan trọng của AXIOM", "theo ngữ nghĩa". Câu cuối trỏ tới III.b (cách giải quyết) và V.a (kết quả) |
| II.a | Kết thúc sau bốn yêu cầu | Thêm câu cuối trỏ tới phương pháp đánh giá ở V.a.1. Đoạn "Bốn yêu cầu này cũng là tiêu chí…" đã bị bỏ theo v4 |
| II.b | "Các module và phân công công việc" với năm tiêu đề module | "Quy trình tổng thể và phần việc của em", ba đoạn văn theo hai giai đoạn của Hình 1, sửa lại chú thích KDL |
| II.c | Là II.b.6 | Trả về II.c đúng mẫu. Dòng của Khôi và đoạn phân công theo v4. Lời cam đoan trỏ về II.c |
| III.a | 6 mục con (đơn vị truy xuất, chunking, hybrid, DISRetrieval, visual, dữ liệu và thước đo) | 3 mục con: Retrieval trên văn bản (gộp ba mục đầu, rút gọn), DISRetrieval, Visual và reranking. Dữ liệu và thước đo chuyển sang V.a.1 |
| III.a.2 DISRetrieval | Có ví dụ F1 trên QASPER và bốn điểm yếu | Theo v4: bỏ ví dụ F1 và điểm yếu thứ tư. Sửa "tốt nhất và Ở" thành "tốt nhất. Ở", sửa "Tuy nhiên, Discourse Parser sử dụng" thành "Trong khi đó, discourse parser này" |
| III.b | "Giải pháp đề xuất và kết quả thực nghiệm", 9 mục con trộn phương pháp và bảng kết quả | "Cách giải quyết của em", 3 mục con chỉ có phương pháp: quy trình cơ sở (Hình 2), cải tiến chất lượng (SEP, visual, reranking trong một mạch), mã hoá theo yêu cầu |
| III.b | "SEP là đóng góp chính của em", "SEP được thiết kế từ kết quả chẩn đoán lỗi" | Bỏ. Mỗi cơ chế trình bày như một phần của quy trình |
| III.c | Có số liệu 2,08 và 1,42 điểm, câu "Báo cáo cũng đo được ngưỡng…" | Chỉ so sánh định tính, sửa tham chiếu |
| IV | Ba mục a, b, c | Hai mục. Câu về Colab và Kaggle gộp vào cuối IV.b |
| V.a | Bốn gạch đầu dòng tóm tắt | 4 mục con: Dữ liệu và phương pháp đánh giá, Lựa chọn chiến lược chunking (Bảng 2), Chất lượng retrieval (phân tích lỗi, Bảng 3 đến 5), Chi phí và thời gian chạy (Bảng 6, 7). Kết thúc bằng đoạn "Tổng hợp lại" |
| V.a.3 | Câu in đậm "nút thắt của chất lượng câu trả lời nằm ở mô hình sinh…", câu "đạt mức cạnh tranh với các giải pháp tốt hiện nay" | Câu thường "với mô hình sinh hiện tại, chất lượng câu trả lời phụ thuộc vào mô hình sinh nhiều hơn vào retrieval". Bỏ câu "cạnh tranh" |
| V | a Kết quả, b Kỹ năng, c Hạn chế, d Hướng phát triển | a Kết quả, b Hạn chế, c Hướng phát triển, d Kỹ năng |

Đã kiểm tra: mọi tham chiếu "mục X.y.z" trỏ đúng tiêu đề, số Bảng và Hình tăng dần theo thứ tự xuất hiện.

### Còn tồn tại
- Hình 2 vẫn là ảnh cũ, còn chữ "ngoại tuyến", "trực tuyến", "Nhánh thị giác".
- Bảng 2, dòng ISE Challenge chờ Nam xác nhận.
- Phần nội dung khoảng 20 trang, mẫu yêu cầu 10 đến 15 trang.

## Bước 8 — Sửa lỗi logic sau khi rà lại toàn bài (2026-09-28)

| Mục | Trước | Sau |
|---|---|---|
| III.a.1 | "Retrieval trên văn bản": chọn trang làm đơn vị, chê chunk quá ngắn, rồi mới nói về chunking, sau đó MaxP, hybrid, hai loại tổn thất | "Retrieval văn bản ở cấp trang". Mạch mới: đơn vị trả kết quả là trang, hai cách chấm điểm văn bản (BM25, dense), giới hạn của dense dẫn tới đơn vị lập chỉ mục là chunk (bốn chiến lược), MaxP quy về trang, hybrid gộp hai nhánh. Công thức MaxP đổi ký hiệu chunk từ p sang x để không trùng với trang p ở MaxSim |
| III.a.1 và V.a.1 | Định nghĩa hai loại tổn thất nằm trong phần lý thuyết. Câu "Recall@10… giúp phân biệt tổn thất tìm kiếm với tổn thất xếp hạng" | Định nghĩa chuyển sang V.a.1, ngay trước Recall@10. Sửa lại rằng phải so Recall@10 với tỉ lệ trang đúng trong pool 100 trang thì mới tách được hai loại tổn thất |
| III.a.2 | Câu hỏi về Bisection không có câu trả lời | Trả lại câu "DISRetrieval cũng luôn vượt Bisection, cho thấy lợi ích đến từ chính quan hệ logic giữa các câu" |
| III.b.2 | "Loại thứ nhất là tổn thất xếp hạng" (thuật ngữ chưa được định nghĩa) | "Loại thứ nhất là xếp sai trang" |
| III.c | "Điểm khác thứ hai…" không có điểm thứ nhất. Hai câu nêu kết quả không có tham chiếu | "Ngoài ra, … (mục V.a.3)", thêm "(mục V.a.3)", "Kết quả ở mục V.a cho thấy" |
| V.a.1 và V.a.2 | Hai mục cùng giới thiệu bộ dữ liệu cho chunking | Mô tả ba bộ dữ liệu gom về V.a.1. V.a.2 ghi rõ Recall@10 ở Bảng 2 được tính trên đơn vị bằng chứng riêng của từng bộ dữ liệu |
| V.a.4 | Không giải thích vì sao Bảng 6 ghi 51,63 còn Bảng 3 ghi 51,73 | Thêm câu: Bảng 6 chạy nhánh visual một mình |
| V.a.4 và đoạn tổng hợp | "Nhanh hơn khoảng 14 lần", nhưng T_first so với quy trình văn bản còn chất lượng so với visual toàn kho | So đúng cặp: 40 phút 29 giây xuống 5 phút 51 giây, khoảng 6,9 lần. Con số 14 lần giữ lại và ghi rõ là so với quy trình cơ sở chỉ dùng văn bản |
| Bảng 7, V.a.4 | "Quy trình gốc" | "Quy trình cơ sở" |
| V.c | "retrieval hiện không còn là nút thắt" | "Với mô hình sinh hiện tại, chất lượng câu trả lời phụ thuộc chủ yếu vào mô hình sinh" |

## Bước 9 — III.a chỉ còn là phần nền tảng (2026-09-28)

Nguyên tắc: III.a chỉ định nghĩa và giải thích cách các phương pháp hoạt động. Lựa chọn thiết kế của nhóm chuyển sang III.b.

| Mục | Trước | Sau |
|---|---|---|
| III.a.1 | "Retrieval văn bản ở cấp trang", mở đầu bằng lựa chọn trang làm đơn vị và câu "đơn vị lập chỉ mục thường nhỏ hơn đơn vị trả kết quả" | "Retrieval văn bản". Định nghĩa IR (hàm điểm và chỉ mục), hai họ phương pháp (lexical với BM25, dense với embedding và cosine), chunk là gì và vì sao cần, bốn chiến lược chunking, MaxP như một cách gộp điểm chung, hybrid và chuẩn hoá |
| III.b.1 | Mở đầu bằng "Quy trình retrieval cơ sở được xây dựng…" | Mở đầu bằng lựa chọn trả kết quả ở cấp trang, hàm s_page và lý do chọn trang (chuyển từ III.a.1) |
| III.a.2 | "Tuy nhiên, khi áp dụng vào bài toán của AXIOM, phương pháp vẫn có ba điểm yếu", "chi phí này làm T_first tăng mạnh" | "Tuy nhiên, phương pháp có ba điểm yếu", "chi phí này tăng theo số tài liệu trong kho" |

## Bước 10 — Rà soát văn phong, thuật ngữ và tính nhất quán số liệu (2026-09-30)

Bản vào: `Tran_Minh_Nam_23021646_Bao_cao_thuc_tap_ban_sua.docx`
Bản ra: `Tran_Minh_Nam_23021646_Bao_cao_thuc_tap_FINAL.docx`

Sửa tại chỗ trên file docx, giữ nguyên toàn bộ định dạng, 3 hình, 8 bảng và mục lục. Nhãn in đậm và ký hiệu in nghiêng được giữ nguyên. Không sửa số liệu bên trong bảng. Tổng cộng 87 vị trí.

Ba mục được yêu cầu giữ nguyên: "khâu sinh câu trả lời" và "khâu sinh phản hồi" (không đổi thành "khâu sinh văn bản"), "slides thuyết trình", "các nút nội tại".

### Lỗi chính tả và trình bày

| Mục | Trước | Sau |
|---|---|---|
| III.a.1 | "có thẻ bắt trọn ngữ nghĩa trừu tượng" | "biểu diễn được quan hệ ngữ nghĩa ở mức trừu tượng" |
| I.b | "bài toàn của hệ thống" | "bài toán của hệ thống" |
| V.a.3 | "bóc tách chính xác phục và chuyển thành thành thành phần có thể được retrieve" | "bóc tách và phục hồi chính xác thành những thành phần có thể truy xuất" |
| V.a.1, Bảng 4 | "tỉ lệ trang đúng", "Tỉ lệ trang đúng" | "tỷ lệ trang đúng", "Tỷ lệ trang đúng" |
| V.b | "Giải pháp hiện tại có 4 hạn chế chính" | "Giải pháp hiện tại còn bốn hạn chế chính" |
| V.a.3 | Chuỗi LaTeX chưa dịch `$A_{\text{file}}$` lọt vào thân văn | `A_file` |
| Mục lục, I.b | "b. Giới thiệu qua bài toán" | "b. Giới thiệu bài toán" |
| V.c | "Thay thế mô hình Visual mã nguồn mở thương mại" (tự mâu thuẫn) | "Thay thế mô hình thị giác bằng mô hình có giấy phép thương mại" |
| II.c | Câu dẫn Bảng 1 thiếu dấu chấm | Thêm dấu chấm |
| V.c | "ở những bước mã hoá ở sau" (câu cụt) | "ở các lần mã hoá sau." |

### Thuật ngữ

| Mục | Trước | Sau |
|---|---|---|
| III.a.1, III.b.1 | "phân mảnh", "phân mảnh nhỏ (chunks)", "Phân mảnh dựa theo", "khâu phân mảnh", "các phân đoạn văn bản" | "chia đoạn", "đoạn nhỏ (chunk)", "Chia đoạn dựa theo", "khâu chia đoạn", "các chunk" |
| III.a.1 | "khoảng chồng lấn (overlap)" | "độ chồng lấn (overlap)" |
| III.a.1 | "tần suất nghịch đảo của từ trong toàn bộ kho dữ liệu (IDF)" | "nghịch đảo tần suất tài liệu (IDF)" |
| III.a.1 | "đo bằng khoảng cách cosine" | "đo bằng độ tương đồng cosine" |
| I.a | "không tìm thấy ứng viên (candidate loss)" | "suy giảm tập ứng viên (candidate loss)" |
| III.a.3 | "cơ chế tương tác muộn (late interaction)" | "cơ chế late interaction" |
| III.a.3 | "các mô hình thị giác ngôn ngữ" | "các mô hình thị giác-ngôn ngữ (vision-language model)" |
| I.b | "tài liệu PDF kỹ thuật số" | "PDF có lớp văn bản" |
| I.b, III.b.2 | "làm thất thoát hoặc biến dạng", "Lỗi thất thoát thông tin trực quan" | "làm mất hoặc biến dạng", "Lỗi mất thông tin trực quan" |
| III.b.2, V.c | "tài liệu có số trang quá dài", "tài liệu dung lượng lớn" | "tài liệu có nhiều trang" |
| V.b | "mất dần tác dụng phân loại" | "mất dần khả năng phân biệt" |
| I.a, III.a.1, III.a.2 | Gạch nối dùng như gạch ngang trong "(Information Retrieval - IR)", "(Structural Evidence Propagation - SEP)", "(Rhetorical Structure Theory - RST)" | Thay bằng dấu phẩy |
| I.a, V.a.3 | "module phân mảnh - nhúng vector", "đầu - cuối (End-to-End)" | "module chia đoạn và nhúng vector", "end-to-end" |

### Ký hiệu toán

| Mục | Trước | Sau |
|---|---|---|
| III.a.3 | `MaxSim(q, p) = Σ_{t ∈ q} max_{v ∈ p} ⟨q_t, v⟩`, công thức dùng `v` nhưng danh sách định nghĩa `v_v` | `MaxSim(q, p) = Σ_{i=1}^{n} max_{j=1..m} ⟨q_i, p_j⟩` |
| III.a.3 | Bảy mục định nghĩa q, p, t, q_t, v, v_v, `<q_t, v_v>` dùng "mảnh ảnh", ký hiệu không khớp công thức | Viết lại thành q, p, i, q_i, j, p_j, ⟨q_i, p_j⟩ dùng "vùng ảnh", khớp công thức |
| III.a.1 | `s_page(q, c) = max_{x ∈ c}`, ký hiệu c trùng nghĩa với ứng viên ở SEP | `s_page(q, p) = max_{x ∈ p}`, thêm chú thích x là chunk thuộc trang p |
| III.a.1, III.b.1 | `alpha` viết bằng chữ | `α` |
| II.a, V.a.1 | `t_q` và `T_q` dùng lẫn lộn | `T_q` |
| II.a | "trung bình thời gian phản hồi truy vấn đầu (T_first)" (T_first là một giá trị, không phải trung bình) | "thời gian tới phản hồi đầu tiên (T_first)" |
| II.b, III.b.3, V.a.4 | `Kf` và `K_f` dùng lẫn lộn, 10 vị trí | `K_f` |

### Văn phong

| Mục | Trước | Sau |
|---|---|---|
| I.a | "Mục tiêu cốt lõi", "câu trả lời luôn đi kèm", "chính là điểm quan trọng", "không thể truy rõ nguồn gốc", "bằng chứng xác đáng nhất" | "Mục tiêu", "câu trả lời đi kèm", "có vai trò then chốt", "không xác định được nguồn gốc", "bằng chứng phù hợp nhất" |
| I.b | "Đáng chú ý, các thông tin", "nhiều định dạng đa dạng" | "Các thông tin", "nhiều định dạng" |
| I.b | "Hệ thống không thể xử lý trước toàn bộ kho dữ liệu một cách tốn kém nếu muốn đáp ứng thời gian đưa vào vận hành thực tế" (trạng ngữ đặt sai làm đảo nghĩa) | "Do đó, hệ thống không thể xử lý trước toàn bộ kho dữ liệu, vì chi phí này khiến thời gian đưa vào vận hành thực tế không còn đáp ứng được" |
| II.a | "đo lường chuẩn hóa trên tập đánh giá ViDoRe V3" | "đo trên tập đánh giá ViDoRe V3 theo cùng một quy trình" |
| III.a.2 | Câu 70 từ liệt kê ba nhược điểm sau dấu hai chấm, có "cực kỳ đắt đỏ" | Tách thành ba câu "Thứ nhất… Thứ hai… Thứ ba…", bỏ "cực kỳ đắt đỏ" |
| III.a.3 | "một mô hình truy xuất thị giác chuyên biệt và hàng đầu hiện nay" | "một mô hình truy xuất thị giác theo cơ chế late interaction" |
| III.a.3 | "kỹ thuật chấm điểm lại (Reranking) sử dụng mô hình cross-encoder thường được sử dụng" | "reranking bằng mô hình cross-encoder thường được áp dụng" |
| III.b.1 | "nhằm tối ưu hóa tính truy vết nguồn gốc bằng chứng" | "nhằm bảo đảm khả năng truy vết nguồn gốc bằng chứng" |
| III.b.2 | "Để giải quyết triệt để hai dạng lỗi này" | "Để giảm hai dạng lỗi này" |
| III.b.2 | "Ý tưởng chính là các trang gold evidence sẽ thường nằm cạnh nhau, chúng ta sẽ gán một hệ số điểm… cho hàng xóm của mỗi trang" (đổi ngôi sang "chúng ta", thì tương lai, calque "hàng xóm") | "SEP dựa trên giả thuyết rằng các trang chứa bằng chứng thường thuộc cùng một tài liệu hoặc nằm gần nhau… thuật toán tính một điểm cấu trúc từ các trang cùng tài liệu và các trang lân cận" |
| III.b.3 | "Để loại bỏ nút thắt này" | "Để giảm nút thắt này" |
| III.c | "nhưng ưu việt hơn ở chỗ tái sử dụng" (so sánh không có đối chứng thực nghiệm) | "nhưng khác ở chỗ tái sử dụng" |
| V.a.1 | "bộ benchmark ViDoRe V3" | "bộ dữ liệu đánh giá ViDoRe V3" |
| V.d | "em đã nắm vững toàn diện", "Rút ra bài học thực nghiệm sâu sắc" (thiếu chủ ngữ), "em làm chủ kỹ năng xây dựng phần mềm chuẩn công nghiệp… monorepo phức tạp" | Viết lại cả mục theo hướng mô tả việc đã làm: "em nắm được các thành phần…", "Em cũng rút ra được rằng cải thiện chỉ số truy xuất chưa chắc chuyển thành độ chính xác cao hơn…", "em thực hành xây dựng phần mềm theo kiến trúc module plugin trong một monorepo" |

### Mạch lập luận

| Mục | Trước | Sau |
|---|---|---|
| I.a | "hai giai đoạn chính… hai module cốt lõi thuộc giai đoạn đầu", mâu thuẫn với Hình 1 vì chunking-embedding và retrieval thuộc Giai đoạn 2 | "hai khâu chính là truy xuất bằng chứng (retrieval) và sinh văn bản dựa trên bằng chứng (generation)… hai module thuộc khâu truy xuất". Từ "giai đoạn" chỉ còn dùng cho cách chia của Hình 1 |
| I.b | "chất lượng của câu trả lời cuối cùng phụ thuộc trực tiếp vào độ chính xác và độ phủ của quá trình truy xuất", bị chính Bảng 5 và hạn chế 4 phủ nhận | "độ chính xác và độ phủ của quá trình truy xuất là điều kiện cần cho chất lượng của câu trả lời cuối cùng" |
| III.a.3 | Mở đầu không nối với kết luận của III.a.2, dùng "ra đời" | Thêm câu chuyển tiếp từ hạn chế của DISRetrieval đối với bảng biểu và hình ảnh sang truy xuất tài liệu thị giác |
| III.a.1 | Một trích dẫn Barnett [11] dùng cho cả semantic chunking lẫn hierarchical chunking | Tách câu, [11] chỉ còn gắn với semantic chunking. Phần hierarchical chunking viết thành nhận định của báo cáo |
| V.a.3 | "Như đã nói ở phần III.b.2… Và đây là các căn cứ… như bàn ở phần III.b.2" (tham chiếu vòng, mở câu bằng "Và") | "Phân tích lỗi trên quy trình truy xuất cơ sở cho thấy hai vấn đề chính. Thứ nhất… Thứ hai… Hai quan sát này là căn cứ thực nghiệm cho thiết kế của SEP và nhánh thị giác đã trình bày ở mục III.b.2" |
| V.a.4 | Đoạn tổng hợp không nối được hai kết luận đối lập ở III.b.2 và V.a.3 | Thêm câu: "Ở mức toàn hệ thống, mức cải thiện của khâu truy xuất chưa chuyển thành độ chính xác cao hơn của câu trả lời, cho thấy khâu sinh văn bản là yếu tố giới hạn tiếp theo" |
| V.c | "dưới đây là những hướng phát triển tiếp" (câu cụt) | "báo cáo đề xuất bốn hướng phát triển tiếp theo." |

### Số liệu

| Mục | Trước | Sau |
|---|---|---|
| I.a, kết quả 1 | "cắt giảm từ 46,0% đến 87,5% số lượng vector… mà không làm suy giảm độ chính xác", mâu thuẫn Bảng 2 vì ISE Challenge giảm 1,7% | "…trong khi Recall@10 tăng trên QASPER và UIT-ViQuAD và chỉ giảm 1,7% trên ISE Challenge" |
| I.a, kết quả 4 | "rút ngắn thời gian khởi tạo phản hồi đầu tiên từ xấp xỉ 40 phút xuống còn dưới 6 phút", ghép chỉ số của hai cấu hình khác nhau và không nói là mô phỏng | "Theo mô phỏng dựa trên các hằng số đo trên GPU NVIDIA L4, nhánh thị giác ở chế độ này với K_f = 3 đạt 50,83 điểm NDCG@10 và rút ngắn…" |
| I.a, kết quả 2 | "chứng minh rằng sự sụt giảm hiệu năng chủ yếu bắt nguồn từ" | "cho thấy phần lớn suy giảm hiệu năng bắt nguồn từ" |
| III.b.2 | "98,3%… nằm trong tập ứng viên 100 trang" gọi là Recall@100, và "Điều này khẳng định" | Gọi đúng tên là Hit@100. "Điều này cho thấy phần lớn suy giảm chất lượng đến từ khâu xếp hạng hơn là từ khâu lấy ứng viên" |
| III.b.1 | "512 từ, chồng lấn 128 từ (được lựa chọn thông qua thực nghiệm tại mục V.a.2)", nhưng V.a.2 không khảo sát kích thước hay độ chồng lấn nào khác | "(cấu hình này cho kết quả không thấp hơn semantic chunking trong thực nghiệm ở mục V.a.2)" |
| III.b.2 | "do số liệu phân tích chứng minh độ tương quan đồng tài liệu mạnh hơn đáng kể" | "do khảo sát cho thấy tương quan giữa các trang cùng tài liệu mạnh hơn tương quan giữa các trang lân cận" |
| III.c | "hoàn toàn không phát sinh chi phí… và áp dụng được cho mọi định dạng tài liệu", mâu thuẫn hạn chế 2 ở V.b | "không phát sinh chi phí tính toán ngoại tuyến và không phụ thuộc vào định dạng gốc của tài liệu" |
| V.a.1 | "kết quả nằm trong khoảng từ 0 đến 1; báo cáo nhân với 100 cho dễ đọc… cao hơn hẳn" | "…và được trình bày theo thang 100. Chẳng hạn… cao hơn" |
| V.a.2 | "fixed-size chunking vượt trội hoàn toàn", "chỉ giảm nhẹ không đáng kể 1,7%", "chứng minh tính đúng đắn" | "tiết kiệm chi phí lưu trữ rõ rệt", "và thấp hơn 1,7%", "do đó fixed-size chunking được chọn làm cấu hình mặc định" |
| V.a.3 | "độ trễ truy vấn trực tuyến chỉ tăng thêm 1,6 ms" | Bỏ câu này. Con số không khớp với hằng số MaxSim 0,74 ms/trang trên pool 100 trang, cần đo lại |
| V.a.3 | "Nhánh visual đóng vai trò quyết định khi đóng góp tới 19,1% cải thiện", nhầm mức tăng tương đối với tỷ trọng đóng góp | "Nhánh thị giác đơn lẻ nâng NDCG@10 thêm 19,1% so với hệ thống cơ sở, chiếm khoảng 85% mức cải thiện của cấu hình kết hợp" |
| V.a.3 | "làm bùng nổ độ trễ trực tuyến lên thêm 700 ms" | "làm độ trễ trực tuyến tăng thêm khoảng 700 ms" |
| V.a.3 | "SEP đạt hiệu quả rõ rệt nhất trên các tập tài liệu có độ dài dưới 50 trang", suy luận nhân quả và xác lập ngưỡng từ năm điểm dữ liệu | "xu hướng SEP hiệu quả hơn trên các tập tài liệu ngắn…" và thêm "Với năm tập con, đây là xu hướng quan sát được chứ chưa đủ để xác lập một ngưỡng cụ thể" |
| V.a.3 | "tỷ lệ câu trả lời chính xác hoàn toàn chỉ tăng nhẹ từ 50,17% lên 50,50%", bỏ qua việc SEP đơn lẻ đạt 51,83% cao hơn cấu hình đề xuất | "gần như không đổi giữa ba cấu hình, lần lượt là 50,17%, 51,83% và 50,50%. Đáng lưu ý, cấu hình SEP đơn lẻ cho tỷ lệ cao nhất dù NDCG@10 thấp hơn cấu hình kết hợp, và chênh lệch… nằm trong mức dao động của trình chấm" |
| V.a.3 | "mô hình sinh chỉ cần khoảng 2 trang để trả lời chính xác" (không có nguồn) | "mô hình sinh chỉ cần một phần nhỏ trong số đó để trả lời đúng" |
| V.a.3 | "tăng vọt thêm 27,73% tuyệt đối (+55,3% tương đối)", đơn vị sai vì 27,73 là điểm phần trăm | "tỷ lệ chính xác hoàn toàn tăng 55,3% tương đối" |
| V.a.3 | "Phát hiện thực nghiệm quan trọng này khẳng định: … nút thắt hiệu năng nằm ở" | "Kết quả này cho thấy… chất lượng câu trả lời bị giới hạn bởi năng lực suy luận của mô hình sinh nhiều hơn bởi khâu truy xuất" |
| V.a.4 | "thời gian lập chỉ mục ngoại tuyến kéo dài của ColVec-8b (chiếm 1h 58m)", nhưng 1h 58m là tổng chi phí của cả hệ thống | "Để giảm thời gian mã hóa toàn kho của ColVec-8b, vốn nâng tổng chi phí ngoại tuyến của hệ thống lên 1h 58m" |
| Bảng 6 | "Chất lượng nhánh Visual (ColVec-8b) theo số lượng tài liệu giữ lại Kf", không nói rõ là chạy độc lập nên dễ hiểu nhầm là mâu thuẫn với 51,73 ở Bảng 3 | "Chất lượng nhánh thị giác (ColVec-8b) chạy độc lập theo số tài liệu giữ lại K_f" |
| V.a.4 | "tiệm cận sát nút mức 51,63", "không mang lại sự khác biệt có ý nghĩa thống kê" (không có kiểm định nào được thực hiện) | "thấp hơn khoảng 1,5% so với khi mã hóa toàn bộ kho tài liệu", "NDCG@10 chỉ dao động trong khoảng 50,83 đến 51,89 điểm và không cho thấy xu hướng tăng rõ rệt" |
| V.a.4 | "trạng thái warm cache đạt tốc độ xử lý vượt trội", "Tốc độ này hoàn toàn có thể tiếp tục được tối ưu hóa" | "ở trạng thái warm cache độ trễ trung vị p50 là 0,16 giây", "Độ trễ ở phân vị đuôi có thể giảm thêm nhờ suy luận theo lô (batching) và quantization, nội dung này thuộc hướng phát triển ở mục V.c" |
| V.a.4 | Không nói rõ mặt trái của on-demand | Thêm: "cơ chế on-demand không làm giảm tổng chi phí mã hóa khi hệ thống phục vụ hết bộ câu hỏi, mà dời chi phí đó từ giai đoạn ngoại tuyến sang các truy vấn đầu tiên" |
| V.a.4 | "cao hơn quy trình cơ sở 7,38 điểm" | "cao hơn quy trình cơ sở 17,0%" |
| V.b | "Cải thiện vượt trội của khâu truy xuất" | "Cải thiện của khâu truy xuất" |

Theo yêu cầu, mọi mức chênh lệch trong thân văn được quy về phần trăm tương đối. Giữ nguyên giá trị đo được đọc từ bảng (ví dụ 43,45 và 53,15 điểm) và các mốc thời gian (700 ms, 0,3 ms, 5 phút 51 giây), vì quy hai loại này sang phần trăm sẽ mất nghĩa. Số liệu trong bảng không sửa.

### Chín mục chưa sửa, cần số liệu gốc

| Mục | Vấn đề |
|---|---|
| Bảng 5 | Cả sáu giá trị khớp chính xác với mẫu số 301 (151, 156, 152, 274, 270, 282 trên 301) và không giá trị nào biểu diễn được với mẫu số 302. Tiêu đề bảng đang ghi 302 câu hỏi |
| Bảng 3 và Bảng 4 | Mức tăng của SEP trên cùng tập physics ghi +2,08 ở Bảng 3 và +2,02 ở Bảng 4. Ngoài ra 45,52 trừ 43,45 bằng 2,07 và 47,42 trừ 43,45 bằng 3,97 |
| Bảng 3 | Nhánh thị giác ghi thêm 1,3 ms mỗi truy vấn, trong khi MaxSim 0,74 ms/trang trên pool 100 trang đã là khoảng 74 ms |
| Bảng 4 | Cột "Tỷ lệ trang đúng trên độ dài tài liệu" chưa định nghĩa cách tính, không đối chiếu được với 7,2 trên 39,9 trang ở V.a.1 |
| V.a.3 | Thực nghiệm GPT-5.2 chưa có bảng, chưa nêu số câu hỏi và trình chấm |
| V.a.3 | Phân tích lỗi chưa có tỷ lệ cho hai dạng lỗi xếp hạng |
| V.a.1 | Không rõ Bảng 3 dùng kết quả parsing từ pdf-inspector hay KDL. Mục II.b ghi ingestion dùng KDL, mục IV.b ghi lệnh chạy với `--parse pdf-inspector` |
| III.b.2 | Không nêu tập dữ liệu dùng để chọn λ và β. Nếu chọn trên chính tập physics thì kết quả Bảng 3 và Bảng 4 có rò rỉ |
| III.a.1 | Cần đối chiếu nội dung bài báo Barnett [11] để xác nhận phạm vi trích dẫn |

### Ghi chú khác

- KDL chưa được viết đầy đủ tên, chưa có trích dẫn và chưa giải thích ở bất kỳ đâu, dù là thành phần parsing của toàn bộ thực nghiệm.
- Hình 2 đặt các khối MaxP, α-fuse và Pool 100 trang ở phía ngoại tuyến, nhưng ba thao tác này chạy theo từng truy vấn.
- Hình 2 ghi nhánh cross-encoder là "top-20", còn mục III.b.1 nói tập ứng viên là 100 trang. Thân báo cáo không nói reranker chỉ chấm lại 20 trang đầu.
- Mười một tài liệu trong danh mục không được trích trong thân bài: [3], [7], [9], [10], [13], [15], [16], [17], [18], [21]. Ngược lại Nemotron và ColQwen2-2B xuất hiện trong Bảng 3 mà không được giới thiệu.

## Bước 11 — Tách Hình 2 thành hai hình (2026-09-30)

Vấn đề: Hình 2 nằm trong mục III.b.1 "Quy trình truy xuất cơ sở", nhưng nội dung hình lại vẽ cả SEP, nhánh thị giác và cross-encoder, tức là ba cơ chế chỉ được giới thiệu ở mục III.b.2 và về sau. Mục III.b.1 không có hình nào mô tả đúng quy trình cơ sở, còn người đọc thì gặp ba khối chưa được định nghĩa.

Lỗi thứ hai trong hình gốc: đường phân cách "ngoại tuyến · một lần" đặt dưới khối Pool 100 trang, khiến MaxP, α-fuse và việc chọn pool bị xếp vào vùng ngoại tuyến. Ba thao tác này chạy theo từng truy vấn.

Cách xử lý: tách thành hai hình, vẽ lại cả hai bằng SVG rồi xuất PNG, dùng đúng bảng màu và font của hình gốc (nền trắng, ô vàng `#F7D66E`, ô xám `#D9D9D9`, viền đen 2px, chú thích đỏ, font Arial). Cả hai đặt ở khổ rộng 14,00 cm như Hình 2 cũ.

| Mục | Trước | Sau |
|---|---|---|
| III.b.1, câu dẫn | "Kiến trúc tổng thể được biểu diễn tại Hình 2." | "Kiến trúc của quy trình cơ sở được biểu diễn tại Hình 2." |
| III.b.1, hình | Hình có đủ SEP, nhánh thị giác, cross-encoder | Hình mới chỉ vẽ quy trình cơ sở: Trang tài liệu → Chunking → chỉ mục BM25 và chỉ mục dense → MaxP → α-fuse → Pool 100 trang → Top-10 trang. Nhánh truy vấn đi vào MaxP. Khổ 14,00 × 12,49 cm |
| III.b.1, chú thích | "Hình 2. Kiến trúc thành phần retrieval chia thành giai đoạn offline và online" | "Hình 2. Kiến trúc quy trình truy xuất cơ sở, chia thành giai đoạn ngoại tuyến và trực tuyến" |
| III.b.2, cuối mục | Không có hình | Thêm câu dẫn "Kiến trúc sau khi bổ sung hai cơ chế được biểu diễn tại Hình 3.", hình mới khổ 14,00 × 14,41 cm, và chú thích "Hình 3. Kiến trúc retrieval sau khi bổ sung SEP và nhánh thị giác". Đặt sau phần mô tả Cơ chế 2, trước mục III.b.3 |

Ba khác biệt so với hình gốc:

- Đường phân cách ngoại tuyến chuyển lên ngay dưới hai chỉ mục. MaxP, α-fuse, Pool 100 trang và khối Truy vấn đều nằm ở vùng trực tuyến.
- Nhãn `recall@100 ≈ 98%` đổi thành `hit@100 ≈ 98%`, khớp với phần thân bài đã sửa ở Bước 10. Đại lượng này đo tỷ lệ câu hỏi có ít nhất một trang bằng chứng trong pool, không phải Recall.
- Hình 2 có thêm nhãn `top-10 theo điểm α-fuse` trên mũi tên cuối, để nói rõ quy trình cơ sở trả kết quả trực tiếp từ pool mà không qua bước xếp hạng lại.

Tệp nguồn của hai hình: `baseline.svg` và `improved.svg`. Ảnh cũ vẫn nằm trong gói docx dưới tên `word/media/image2.png` nhưng không còn được tham chiếu.
