# Báo cáo tiến độ rl-lab

Cập nhật: 2026-09-25. Mô hình học (student): Qwen2.5-0.5B-Instruct, 4-bit QLoRA. GPU: RTX 5060 Ti 16 GB, chạy trong Docker.

## 1. Kết quả hiện có

Đánh giá chung cho mọi mô hình: 1.000 câu đầu của GSM8K test, giải tham lam (greedy), so khớp số cuối cùng.
Dữ liệu huấn luyện: 1.500 dòng cho mỗi thí nghiệm. Các thí nghiệm GSM8K dùng cùng 1.500 câu train (seed 42).

| Phương pháp | Dữ liệu huấn luyện | Độ chính xác | Khoảng tin cậy 95% (xấp xỉ) | Định dạng `#### n` | Độ dài trả lời (token) | Thời gian huấn luyện | Bộ nhớ đỉnh | Batch đánh giá |
|---|---|---|---|---|---|---|---|---|
| Baseline | không huấn luyện | 31,1% | 28,2 đến 34,0 | 0,1% | 256 | n/a | n/a | 96 |
| SFT | `trl-lib/Capybara` (chat tổng quát) | 25,9% | 23,2 đến 28,6 | 0,3% | 205 | 666 giây | 1,66 GB | 96 |
| DPO | `trl-lib/ultrafeedback_binarized` (chat tổng quát) | 30,4% | 27,6 đến 33,3 | 0,0% | 255 | 542 giây | 8,49 GB | 128 |
| SFT-GSM8K | lời giải GSM8K do **người** viết | 30,1% | 27,3 đến 32,9 | 99,2% | 83 | 85 giây | 2,02 GB | 128 |
| Distill (seq) | lời giải GSM8K do **teacher Qwen2.5-1.5B** viết | **32,6%** | 29,7 đến 35,5 | 22,8% | 247 | 142 giây (chưa tính teacher) | 2,02 GB | 128 |

Chi tiết huấn luyện:
- SFT: loss cuối 1,51, 94 bước, batch 2 × 8.
- DPO: loss 0,686 (bắt đầu từ 0,693), `rewards/accuracies` 0,53, `rewards/margins` 0,02, 93 bước, batch 2 × 8, learning rate 5e-6.
- SFT-GSM8K: loss 0,60, 94 bước, batch 16 × 1.
- Distill (seq): loss 0,33, độ chính xác theo token 0,88, 94 bước, batch 16 × 1. Teacher 1.5B trả lời đúng 71,3% trong 1.500 câu train, đúng định dạng 49,1%.

### Nhận xét

1. **SFT trên chat tổng quát làm giảm khả năng toán** (-5,2 điểm, khoảng 2,6 sai số chuẩn, có ý nghĩa thống kê). Câu trả lời ngắn đi (205 so với 256 token), mô hình lập luận ít hơn. Đây là hiện tượng catastrophic forgetting khi dữ liệu lệch nhiệm vụ.
2. **DPO trên UltraFeedback gần như không học được gì.** `rewards/accuracies` 0,53 là gần mức đoán ngẫu nhiên. Nguyên nhân: learning rate 5e-6 quá thấp cho LoRA với 93 bước, và dữ liệu không phải toán.
3. **Phong cách lời giải quan trọng hơn việc đúng định dạng.** SFT trên lời giải của người học định dạng gần hoàn hảo (99,2%) nhưng trả lời rất ngắn (83 token) và không tăng độ chính xác. Lời giải chi tiết của teacher (247 token) cho kết quả cao nhất (32,6%).
4. **Mọi chênh lệch trừ SFT đều nằm trong sai số** (khoảng ±3 điểm với 1.000 câu). Chỉ có một seed cho mỗi cấu hình.

### Lưu ý về độ tin cậy

- Baseline và SFT được đánh giá với batch 96, các lần sau với batch 128. Batch khác nhau có thể làm kết quả greedy lệch rất nhẹ, nên cần đánh giá lại baseline với batch 128.
- Thời gian SFT bị tăng do chạy cùng lúc với các bài kiểm thử nhỏ của agent. SFT cũng huấn luyện với batch 2 × 8, khác với các lần sau.
- Chi phí sinh câu trả lời của teacher (khoảng 11 phút, ước tính từ 96 câu trong 41,6 giây) chưa được ghi vào record distill, vì lần chạy cuối dùng lại câu trả lời đã lưu.
- Giá GPU theo giờ chưa có (`cost.gpu_hourly_usd: null`), nên chi phí chỉ tính bằng giờ GPU.

## 2. Việc đã làm

### Môi trường
- Cài kernel WSL2 để Docker Desktop chạy được với GPU.
- Tạo Docker image `rl-lab:cu128` (CUDA 12.8, hỗ trợ GPU Blackwell), dùng docker compose với GPU, mount repo, lưu cache Hugging Face trong volume `hf-cache`.
- Khóa phiên bản bằng `uv.lock`: torch 2.13.0, trl 1.13.0, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, mlflow 3.16.1.
- `scripts/prefetch.py`: tải trước mô hình và dataset vào cache.

### Sửa lỗi (các giả định chưa kiểm chứng trong CLAUDE.md)
- transformers 5 bỏ `warmup_ratio`: đổi sang `warmup_steps` (giá trị dưới 1 được hiểu là tỷ lệ).
- MLflow 3 không đăng ký được thư mục adapter: đăng ký bằng đường dẫn artifact (`src/lab/tracking.py`).
- Đánh giá: in tiến độ theo batch, batch cấu hình được (`eval.batch_size`), đo thời gian đánh giá và bộ nhớ đỉnh cho baseline.
- Đánh giá bị lỗi khi mô hình sinh số cực lớn (tràn thành vô cực): giờ tính là sai thay vì dừng chương trình, có unit test.
- Giải phóng bộ nhớ GPU sau khi huấn luyện (optimizer, gradient, teacher) và sau khi teacher sinh xong câu trả lời.

### Phương pháp mới
- **Distillation** (`src/lab/distill.py`): cấp chuỗi (SFT trên câu trả lời của teacher) và logit-KL (thêm loss KL với phân phối của teacher). Câu trả lời của teacher được lưu trong `outputs/distill/` và tái sử dụng.
- **GRPO** (`src/lab/grpo.py`): phần thưởng đúng đáp án và phần thưởng định dạng. Đã chạy thử nhỏ thành công, nhưng lần chạy đầy đủ (4 đến 5 giờ) bị bỏ theo quyết định của bạn.
- **SFT trên lời giải GSM8K của người** (`to_gsm8k_solutions` trong `src/lab/data.py`, config `configs/sft_gsm8k_qlora.yaml`).
- Tất cả được đăng ký trong `METHODS` của `src/lab/train.py`. 23 unit test đều đạt.

### Theo dõi và công bố
- Mỗi lần chạy có MLflow run và file record JSON trong `outputs/records/` (git commit, phiên bản mô hình và dataset, config, kết quả, chi phí, bộ nhớ). Record của các lần chạy thử nằm riêng trong `outputs/smoke-records/` (không commit).
- `scripts/push_adapter.py`: tải adapter, model card và record lên Hugging Face (repo private), rồi ghi link và commit Hub vào record.
- Đã tải lên 4 adapter: `quyenpro/rl-lab-{sft-qlora, dpo-qlora, distill-seq-qlora, sft-gsm8k-qlora}-qwen2-5-0-5b-instruct`.
- GitHub `quyen244/rl-llms-lab`: 4 commit, tác giả chỉ có bạn, đã push hết.
- Bảo mật: token Hugging Face từng nằm trong `.env.example` (file được git theo dõi). Đã chuyển sang `.env` (bị git bỏ qua) trước khi commit, token chưa bao giờ bị commit.

### Tài liệu
- `docs/methods-guide.md`: giải thích QLoRA, baseline, SFT, DPO (ý tưởng, công thức, cấu hình, chỉ số cần theo dõi, lỗi thường gặp).

## 3. Sự cố đã gặp

| Sự cố | Hậu quả | Cách xử lý |
|---|---|---|
| Máy tự khởi động lại (01:20, 01:54 và một lần sau đó), do `shutdown.exe` chạy dưới quyền SYSTEM | Mất lần đánh giá DPO đầu tiên, mất lần chạy logit-KL | Chạy lại. Nguyên nhân nằm ngoài dự án (nhà cung cấp cloud hoặc chính sách bảo trì Windows), cần kiểm tra |
| Tắt gradient checkpointing ở batch 16 | Dùng hết 16 GB, tràn sang RAM hệ thống, 10 giây/bước | Bật lại checkpointing: 2 GB, 1,5 giây/bước |
| DPO batch 8 | Hết bộ nhớ GPU | Giữ batch 2 |
| Logit-KL batch 8 | Cần 26 GB, tràn sang RAM | Dùng batch 2 (7,8 GB) |

Trên máy Windows này, khi vượt 16 GB, driver không báo lỗi hết bộ nhớ mà âm thầm dùng RAM hệ thống, làm chương trình chậm đi rất nhiều.

## 4. Việc chưa làm

| Việc | Trạng thái | Thời gian ước tính |
|---|---|---|
| Logit-KL distill | Code sẵn sàng, đã chạy thử, lần chạy đầy đủ bị mất do máy khởi động lại | khoảng 20 phút |
| Đánh giá lại baseline với batch 128 | Chưa làm | khoảng 5 phút |
| SFT-GSM8K rồi DPO trên dữ liệu toán tự tạo (cặp lời giải đúng/sai, learning rate 5e-5, 1.500 câu train mới) | Đã lên kế hoạch, chờ bạn xác nhận | khoảng 1 giờ code và 30 phút chạy |
| Tạo báo cáo so sánh (`lab.report`) và chọn mô hình tốt nhất làm `champion` | Chưa làm | khoảng 10 phút |
| Cập nhật `docs/methods-guide.md` (distill, SFT-GSM8K, kết quả) và CLAUDE.md (trạng thái kiểm chứng) | Chưa làm | khoảng 20 phút |
| Ghi chi phí teacher vào record distill | Chưa làm | nhỏ |

### Đã hoãn hoặc bỏ

- GRPO chạy đầy đủ: bỏ vì quá lâu (4 đến 5 giờ). Code vẫn còn.
- PPO: không cần (bạn muốn DPO).
- Teacher 7B, student 1.5B, nhiều seed, quét beta cho DPO, so sánh QLoRA với LoRA thường, đánh giá IFEval và MATH-500: để sau.
- MLflow registry còn một phiên bản từ lần chạy thử (SFT phiên bản 1, gắn tag `smoke=true`).
