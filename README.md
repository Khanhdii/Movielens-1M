# Colaborative Filtering
Trong bài này, phương pháp Collaborative Filtering (CF) được áp dụng để xây dựng hệ thống gợi ý dựa trên tập dữ liệu MovieLens 1M.

Trong code có mở rộng thêm phần Embedding với lý do: thay vì chỉ chuẩn hóa theo công thức cho movieID và userID. Mô hình có thể sử dụng Embedding để giúp mô hình nắm bắt thêm thông tin, học được các đặc trưng tiềm ẩn từ dữ liệu và cải thiện độ chính xác dự đoán. Lý do của điều này là vì khi embedding, ta có thể tạo được một vector (nhiều hơn 1 chiều) để biểu diễn movieID và userID, từ đó, có nhiều số hạng hơn cho mô bình biến đổi và cập nhật thông tin

![image](https://github.com/user-attachments/assets/330a770e-a5eb-4494-80bc-34d6e7748d1f)

# DSSM
Trong phần DSSM, code được triển khai tham khảo theo link https://github.com/RUCAIBox/RecBole và paper https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf.

Kết quả:
1. Khi triển khai theo ý tưởng của 2 link tham khảo, kết quả thu được loss trong khoảng 0.5
2. Thay vì chỉ phân loại nhị phân (ratings>3 - Thích; rating <= 3 - không thích), code này triển khai thêm phần sử dụng cross entropy cho cả 5 lớp ratings. Kết quả thu được loss 1.176 trên tập validation 
