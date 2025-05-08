Mô hình Dịch Máy Transformer
Dự án triển khai mô hình Transformer cho dịch máy sử dụng PyTorch Lightning.
Cấu trúc dự án

data.py: Xử lý dữ liệu, tạo dataset và dataloader.
config.py: Cấu hình tham số dòng lệnh.
main.py: Điểm bắt đầu để huấn luyện và kiểm tra.
trainer.py: Lớp Trainer tùy chỉnh (không sử dụng trong main.py).
utils.py: Hàm hỗ trợ ghi log và lưu mô hình.
model/:
transformer.py: Mô hình Transformer chính.
encoder.py: Lớp mã hóa.
decoder.py: Lớp giải mã.
attention.py: Cơ chế Multi-Head Attention.
feedforward.py: Mạng FeedForward.
positional_encoding.py: Mã hóa vị trí.


requirements.txt: Danh sách thư viện yêu cầu.

Yêu cầu

Python 3.8+
Cài đặt thư viện:pip install -r requirements.txt



Cài đặt

Clone kho lưu trữ:git clone <repository-url>
cd <repository-directory>


Cài đặt thư viện:pip install -r requirements.txt



Sử dụng
Chuẩn bị dữ liệu
Dữ liệu dạng tệp văn bản, phân tách bằng tab:
câu_nguồn<TAB>câu_đích<TAB>cột_thêm

Huấn luyện
Chạy lệnh:
python main.py --train_file đường/dẫn/train.txt

Các tham số chính:

--train_file: Đường dẫn tệp dữ liệu (bắt buộc).
--max_len: Độ dài tối đa chuỗi (mặc định: 50).
--batch_size: Kích thước batch (mặc định: 512).
--epochs: Số epoch (mặc định: 100).
--learning_rate: Tốc độ học (mặc định: 0.0001).

Theo dõi

TensorBoard: Xem metrics tại logs/:tensorboard --logdir logs/


Checkpoints: Lưu mô hình tốt nhất trong checkpoints/.

Kiểm tra
Mô hình tự động đánh giá trên tập test, tính BLEU score.
Ghi chú

Sử dụng mixed precision để tối ưu GPU.
Dataset chia: 80% train, 10% val, 10% test.
Cần cải thiện BLEU score bằng cách ánh xạ token về từ.

Cải tiến tương lai

Thêm script suy luận.
Tối ưu tham số mô hình.
Tăng cường dữ liệu.

Giấy phép
MIT License (nếu có).
