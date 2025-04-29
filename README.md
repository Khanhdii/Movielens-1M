Dễ hiểu rồi!  
Bạn muốn **đảo ngược thứ tự các phần** theo yêu cầu như sau:

- Bắt đầu từ **Seq2Seq** trước.
- Sau đó đến **MIND (Alibaba)**.
- Tiếp theo là **DSSM**.
- Cuối cùng là **Collaborative Filtering**.

Và giữ nguyên nội dung bạn vừa gửi nhưng trình bày đẹp hơn, thêm chút **màu sắc nhẹ nhàng** cho dễ nhìn (khi in hoặc copy vào Word/LaTeX).

---

Dưới đây là bản đã **đảo ngược và trình bày đẹp**:

---

# 📚 Seq2Seq

## 1. Giới thiệu mô hình
Đề tài thực hiện xây dựng hệ thống dịch máy từ tiếng Anh sang tiếng Pháp sử dụng kiến trúc Seq2Seq cải tiến với các thành phần mới nhằm nâng cao chất lượng bản dịch:
- ✨ **Self-Attention** trong Encoder
- ✨ **Cross-Attention** trong Decoder
- ✨ **Multi-Head Attention**
- ✨ **Beam Search** decoding
- ✨ **Pytorch Lightning** tối ưu quy trình huấn luyện

Dữ liệu sử dụng: **fra.txt** gồm các cặp câu tiếng Anh – tiếng Pháp.

---

## 2. Cấu trúc hệ thống

### 2.1. Encoder
- **Kiến trúc:** BiLSTM hai tầng.
- ✅ Thêm **Self-Attention** sau LSTM.
- ✅ Áp dụng **LayerNorm** và **Dropout**.

### 2.2. Decoder
- **Kiến trúc:** LSTM hai tầng.
- ✅ Thêm **Cross-Attention** từ Encoder.
- ✅ Ổn định đầu ra với **LayerNorm**.

### 2.3. Tổng thể mô hình Seq2Seq
- 🔥 Hỗ trợ dịch bằng:
  - **Greedy Search**
  - **Beam Search**

---

## 3. Quy trình huấn luyện

- **Framework:** PyTorch Lightning.
- **Kỹ thuật huấn luyện:** 
  - 📝 ModelCheckpoint
  - 📝 EarlyStopping
- **Embedding:** 
  - Sử dụng **GloVe 840B** (nếu có).
- **Tối ưu hóa:** Adam + ReduceLROnPlateau.

---

## 4. Cải tiến so với Notebook gốc

| Nội dung | Notebook Cũ | Phiên bản Mới |
|:---|:---|:---|
| Attention | ❌ Không có | ✅ Self-Attention + Cross-Attention |
| Beam Search | ❌ Không áp dụng | ✅ Có |
| Code Organization | ❌ 1 file | ✅ Module hóa |
| BLEU Evaluation | ❌ Thủ công | ✅ Tự động log |
| Training | ❌ Thủ công | ✅ PyTorch Lightning |

---

## 5. Kết quả thực nghiệm

🌟 **Kết quả trên tập kiểm thử:**

| 🎯 Chỉ số | 📊 Giá trị |
|:---|:---|
| **Test BLEU Score** | **0.3074** |
| **Test Loss** | **2.0386** |

---

# 🧠 MIND (Alibaba)

## 1. Giới thiệu mô hình
- **MIND (Multi-Interest Network with Dynamic Routing)** giúp biểu diễn **sở thích đa dạng** của người dùng.
- Cơ chế:
  - **Dynamic Routing:** nhóm hành vi thành cụm sở thích.
  - **Label-aware Attention:** chọn cụm sở thích phù hợp với item mục tiêu.

### Kết quả truyền thống:
![image](https://github.com/user-attachments/assets/4237ad27-c664-45ff-9daf-5b877c4fb897)

---

## 2. Điểm mới & cải tiến
✨ **Tích hợp Multi-Head Attention (MHA)**:
- Thay thế attention truyền thống bằng MHA để học được nhiều góc nhìn hơn giữa sở thích và item mục tiêu.

### Kết quả cải tiến:
![image](https://github.com/user-attachments/assets/4d2c6c7f-ba4b-46b8-9c75-f80ef8459575)

---

## 3. Tóm tắt
- Triển khai đầy đủ mô hình **MIND**.
- Tùy chọn bật/tắt **MHA**.
- Áp dụng **EarlyStopping**, lưu **Checkpoint**.
- Dùng **TensorBoard** theo dõi training/validation.
- Thao tác dễ dàng bằng tham số dòng lệnh (**args**).

---

## 4. Tham số tùy chỉnh

| Tham Số         | Mô Tả                                         | Mặc Định  |
|-----------------|-----------------------------------------------|-----------|
| `--ratings_path`| Đường dẫn file ratings.dat                    | `data/ratings.dat` |
| `--batch_size`  | Batch size                                    | 128       |
| `--seq_len`     | Độ dài hành vi người dùng                     | 5         |
| `--embedding_dim`| Kích thước vector embedding                  | 32        |
| `--num_interests`| Số lượng vector sở thích (K)                 | 4         |
| `--lr`          | Learning rate                                 | 0.001     |
| `--max_epochs`  | Số vòng lặp tối đa                            | 50        |
| `--use_mha`     | Dùng Multi-Head Attention                     | False     |
| `--num_heads`   | Số lượng heads của MHA                        | 2         |

---

## 5. Ghi chú thêm
- **Lần này sử dụng Kaggle Notebook**: vì dễ treo máy, không cần bật local IDE như PyCharm.

---

# 🔥 DSSM

## 1. Giới thiệu mô hình
- Triển khai theo:
  - Link tham khảo: [RecBole DSSM](https://github.com/RUCAIBox/RecBole)
  - Paper gốc: [Deep Structured Semantic Models for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)

---

## 2. Kết quả thực nghiệm
- Khi triển khai theo ý tưởng gốc:
  - Loss đạt khoảng **0.5**.
- Khi phân loại theo **5 mức rating** (multi-class classification bằng cross-entropy):
  - Loss trên tập validation khoảng **1.176**.

---

# 🎬 Collaborative Filtering (CF)

## 1. Giới thiệu mô hình
- Phương pháp **Collaborative Filtering (CF)** được áp dụng trên tập MovieLens 1M để xây dựng hệ thống gợi ý.

---

## 2. Cải tiến

- Thay vì chỉ chuẩn hóa `movieID` và `userID`, dự án đã:
  - Áp dụng **Embedding Layer** cho `movieID` và `userID`.
  - Nhằm mục đích:
    - ✅ Học được các **đặc trưng tiềm ẩn** (latent features).
    - ✅ Biểu diễn ID thành vector nhiều chiều ➔ mô hình dễ học tốt hơn.

### Hình minh họa:
![image](https://github.com/user-attachments/assets/330a770e-a5eb-4494-80bc-34d6e7748d1f)
