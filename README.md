# Colaborative Filtering
Trong bài này, phương pháp Collaborative Filtering (CF) được áp dụng để xây dựng hệ thống gợi ý dựa trên tập dữ liệu MovieLens 1M.

Trong code có mở rộng thêm phần Embedding với lý do: thay vì chỉ chuẩn hóa theo công thức cho movieID và userID. Mô hình có thể sử dụng Embedding để giúp mô hình nắm bắt thêm thông tin, học được các đặc trưng tiềm ẩn từ dữ liệu và cải thiện độ chính xác dự đoán. Lý do của điều này là vì khi embedding, ta có thể tạo được một vector (nhiều hơn 1 chiều) để biểu diễn movieID và userID, từ đó, có nhiều số hạng hơn cho mô bình biến đổi và cập nhật thông tin

![image](https://github.com/user-attachments/assets/330a770e-a5eb-4494-80bc-34d6e7748d1f)

# DSSM
Trong phần DSSM, code được triển khai tham khảo theo link https://github.com/RUCAIBox/RecBole và paper https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf.

Kết quả:
1. Khi triển khai theo ý tưởng của 2 link tham khảo, kết quả thu được loss trong khoảng 0.5
2. Thay vì chỉ phân loại nhị phân (ratings>3 - Thích; rating <= 3 - không thích), code này triển khai thêm phần sử dụng cross entropy cho cả 5 lớp ratings. Kết quả thu được loss 1.176 trên tập validation 4


# MIND (Alibaba)

Phần này trình bày triển khai mô hình **MIND (Multi-Interest Network with Dynamic Routing)** dựa trên paper gốc, áp dụng vào tập dữ liệu **MovieLens-1M** để giải quyết bài toán gợi ý. Bên cạnh đó, dự án còn thực hiện các **cải tiến quan trọng** nhằm nâng cao hiệu quả mô hình hóa sở thích người dùng.

## MIND Truyền Thống
- MIND giúp biểu diễn **sở thích đa dạng** của người dùng thông qua nhiều vector (multi-interest vectors).
- Sử dụng cơ chế:
  - **Dynamic Routing** để nhóm hành vi người dùng thành các cụm sở thích.
  - **Label-aware Attention** để chọn vector sở thích phù hợp với item mục tiêu.
  - Kết quả:
![image](https://github.com/user-attachments/assets/4237ad27-c664-45ff-9daf-5b877c4fb897)


##  Điểm Mới & Cải Tiến: **Tích hợp Multi-Head Attention (MHA)**:
   - Thay thế cơ chế attention truyền thống bằng **Multi-Head Attention**.
   - MHA giúp mô hình học được nhiều góc nhìn (representations) khi liên kết giữa sở thích và item mục tiêu.
   - Kết quả:
![image](https://github.com/user-attachments/assets/4d2c6c7f-ba4b-46b8-9c75-f80ef8459575)


## Tóm tắt
- Triển khai đầy đủ mô hình **MIND** theo paper.
- Tùy chọn bật/tắt **Multi-Head Attention**.
- Áp dụng **EarlyStopping** và lưu checkpoint tốt nhất.
- Xử lý dữ liệu MovieLens theo dạng **sequence behavior**.
- Tích hợp **TensorBoard** để theo dõi train/val loss.
- Tùy chỉnh dễ dàng thông qua các tham số dòng lệnh (**args**).

## Tham Số Tùy Chỉnh

| Tham Số         | Mô Tả                                         | Mặc Định  |
|-----------------|-----------------------------------------------|-----------|
| `--ratings_path`| Đường dẫn tới file ratings.dat                | `data/ratings.dat` |
| `--batch_size`  | Kích thước batch                              | 128       |
| `--seq_len`     | Độ dài sequence hành vi                       | 5         |
| `--embedding_dim`| Kích thước vector embedding                  | 32        |
| `--num_interests`| Số lượng vector sở thích (K)                 | 4         |
| `--lr`          | Learning rate                                 | 0.001     |
| `--max_epochs`  | Số vòng lặp tối đa                            | 50        |
| `--use_mha`     | Kích hoạt Multi-Head Attention                | False     |
| `--num_heads`   | Số lượng heads cho Multi-Head Attention       | 2         |

**Lý do sử dụng Kaggle Notebook cho lần triển khai này:** Em đã quen làm việc trên môi trường local với PyCharm và môi trường ảo để dễ kiểm soát code và thư viện.Lần này em dùng Kaggle để treo máy mà không cần bật code

# Seq2Seq

## 1. Giới thiệu mô hình
Đề tài thực hiện xây dựng hệ thống dịch máy từ tiếng Anh sang tiếng Pháp sử dụng kiến trúc Seq2Seq cải tiến với các thành phần mới nhằm nâng cao chất lượng bản dịch, bao gồm:
- ✨ Self-Attention trong Encoder
- ✨ Cross-Attention trong Decoder
- ✨ Multi-Head Attention
- ✨ Beam Search decoding
- ✨ Pytorch Lightning để tối ưu huấn luyện và kiểm thử

Dữ liệu được sử dụng: **fra.txt** gồm các cặp câu tiếng Anh – tiếng Pháp.

---

## 2. Cấu trúc hệ thống

### 2.1. Encoder
- **Kiến trúc:** BiLSTM hai tầng.
- **Cải tiến:** 
  - ✅ Thêm **Self-Attention** sau lớp LSTM để tổng hợp tốt hơn ngữ cảnh.
  - ✅ Áp dụng **Layer Normalization** và **Dropout** để ổn định quá trình học.

### 2.2. Decoder
- **Kiến trúc:** LSTM hai tầng.
- **Cải tiến:** 
  - ✅ Thêm **Cross-Attention** giúp Decoder lấy thông tin từ Encoder.
  - ✅ Tăng khả năng tổng hợp và tạo ra câu đầu ra chính xác hơn.

### 2.3. Tổng thể mô hình Seq2Seq
- 🔥 Hỗ trợ 2 chế độ dịch:
  - **Greedy Search:** chọn token xác suất cao nhất từng bước.
  - **Beam Search:** lưu trữ nhiều lựa chọn tốt nhất, tối ưu toàn chuỗi.

---

## 3. Quy trình huấn luyện

- **Framework:** Pytorch Lightning
- **Kỹ thuật huấn luyện:** 
  -  **ModelCheckpoint**: lưu mô hình tốt nhất dựa trên `val_loss`.
  -  **EarlyStopping**: dừng sớm nếu `val_loss` không giảm.
- **Embedding:** 
  - Tự động khởi tạo từ **GloVe 840B 300d** nếu có.
- **Tối ưu hóa:**
  - `Adam` Optimizer
  - `ReduceLROnPlateau` Scheduler

---

## 4. Cải tiến so với phiên bản Notebook ban đầu

| Nội dung | Phiên bản Notebook cũ | Phiên bản Mới (Module hóa) |
|:---|:---|:---|
| **Attention** | ❌ Không có hoặc đơn giản | ✅ Thêm Self-Attention và Cross-Attention với Multi-Head |
| **Beam Search** | ❌ Không áp dụng | ✅ Beam Search decoding thông minh |
| **Tổ chức code** | ❌ Tất cả trong 1 file | ✅ Tách thành các module nhỏ gọn dễ quản lý |
| **Đánh giá BLEU** | ❌ Thủ công | ✅ BLEU tự động log và hiển thị |
| **Training** | ❌ Huấn luyện thủ công | ✅ Huấn luyện tự động hóa với Lightning |
| **EarlyStopping** | ❌ Không có | ✅ Thêm EarlyStopping và Checkpoint |
| **Xử lý từ vựng** | ❌ Gán thủ công | ✅ Build từ vựng tự động |
| **Embedding** | ❌ Random | ✅ GloVe pre-trained Embedding |

---

## 5. Kết quả thực nghiệm

🌟 **Sau khi huấn luyện hoàn tất, mô hình đạt kết quả sau trên tập kiểm thử:**

| 🎯 Chỉ số | 📊 Giá trị |
|:---|:---|
| **Test BLEU Score** | **0.3074** |
| **Test Loss** | **2.0386** |


> Với BLEU ~30%, đây là kết quả khả quan cho mô hình Seq2Seq đơn giản hóa, có thể được nâng cao hơn nếu áp dụng các kỹ thuật tiền xử lý và mô hình lớn hơn.
