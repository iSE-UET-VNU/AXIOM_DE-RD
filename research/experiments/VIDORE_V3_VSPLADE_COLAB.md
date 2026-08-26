# ViDoRe v3 Industrial + V-SPLADE trên Colab L4

Thí nghiệm dùng đúng checkpoint local đã kiểm tra (`data/output/vsplade/efficient`),
không tải lại model. Corpus Industrial chứa ảnh PNG đã render sẵn trong parquet,
nên runner không render PDF lần nữa.

## Chuẩn bị một lần trên Google Drive

Đưa các phần sau của repository lên cùng một thư mục, ví dụ
`MyDrive/AXIOM_DE-RD/`:

```text
research/experiments/vidore_v3_vsplade.py
src/
data/raw/benchmarks/vidore_v3/
data/output/vsplade/efficient/
```

Kích thước hiện tại khoảng 2,5 GB cho dataset và 0,62 GB cho checkpoint.

## Các cell chạy trong hosted Colab

Chọn **L4 GPU**, rồi chạy:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/AXIOM_DE-RD
!pip install -q fastparquet "transformers==5.3.0" "sentence-transformers==5.6.1" "peft==0.20.0"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Benchmark giữ nguyên batch 3 để so sánh trực tiếp với máy local:

```python
!python -u research/experiments/vidore_v3_vsplade.py \
  --dataset-root data/raw/benchmarks/vidore_v3 \
  --model data/output/vsplade/efficient \
  --output-dir data/output/vsplade/vidore_v3_industrial_english_283q \
  --subset industrial \
  --language english \
  --device cuda \
  --batch-size 3 \
  --top-k 100
```

Sau khi đo baseline cùng batch 3, có thể thử batch 8 hoặc 16 trên L4 để đo
trade-off tốc độ/bộ nhớ; kết quả đó nên lưu ở output directory khác.
