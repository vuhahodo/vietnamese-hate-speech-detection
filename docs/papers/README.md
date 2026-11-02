---
permalink: nlp-commerce-doctor/docs/papers/readme
title: README
type: note
---

# Bài báo tham khảo

PDF và bản trích văn bản (`.txt`) chỉ lưu cục bộ, không commit (repo public). Thông tin dưới đây đọc từ PDF ngày 02/11/2026.

## 1. Vai trò từng bài

| Vai trò | Bài | Trạng thái xuất bản | File cục bộ |
|---|---|---|---|
| Dữ liệu + baseline gốc | Co Van Dinh, Son T. Luu, Anh Gia-Tuan Nguyen. *Detecting Spam Reviews on Vietnamese E-commerce Websites.* arXiv 2207.14636 | arXiv; cần kiểm nơi đăng chính thức (thường được trích là ACIIDS 2022) | `arxiv_2207.14636.pdf` |
| Bản mở rộng có metadata | Co Van Dinh, Son T. Luu. *Metadata Integration for Spam Reviews Detection on Vietnamese E-commerce Websites.* International Journal of Asian Language Processing (World Scientific), 2024. arXiv 2405.13292 | Tạp chí | `arxiv_2405.13292.pdf` |
| Mức trần PLM | Nguyen et al. *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing.* EMNLP 2023. arXiv 2310.11166 | Hội nghị | `arxiv_2310.11166.pdf` |
| **Pipeline kế thừa (có phản biện)** | Ala' M. Al-Zoubi, Antonio M. Mora, Hossam Faris. *A Multilingual Spam Reviews Detection Based on Pre-Trained Word Embedding and Weighted Swarm Support Vector Machines.* IEEE Access 11 (2023) 72250-72271. doi:10.1109/ACCESS.2023.3293641 | Tạp chí, truy cập mở | đã tải và đã đọc: `A_Multilingual_Spam_Reviews_Detection.pdf` |
| Pipeline kế thừa, bản mới | Al-Zoubi, Mora, Faris, Qaddoura. *A hybrid TwinSVM-HHO model for multilingual spam review detection using sentiment features and pre-trained embeddings.* Expert Systems with Applications 287 (2025) 128160. doi:10.1016/j.eswa.2025.128160 | Tạp chí | **chưa tải**: https://digibug.ugr.es/handle/10481/110424 |
| Khuôn khối đặc trưng + stacking | Shabbir Anees et al. *An Optimized Machine Learning Classifier for Detecting Fake Reviews Using Extracted Features.* arXiv 2511.21716 | Preprint, chưa phản biện: chỉ dùng làm khuôn | `arxiv_2511.21716.pdf` |
| Phương pháp EA | *Compact NSGA-II for Multi-objective Feature Selection.* arXiv 2402.12625 | Cần kiểm nơi đăng | `arxiv_2402.12625.pdf` |

## 2. Số liệu gốc trên ViSpamReviews

T1 = spam / không spam; T2 = loại spam (4 lớp). Chỉ số %, test set.

**Dinh, Luu, Nguyen (split v1: 14.306 / 1.590 / 3.974, tỉ lệ 7-1-2), Bảng 8:**

| Mô hình | Acc T1 | Acc T2 | Macro F1 T1 | Macro F1 T2 |
|---|---:|---:|---:|---:|
| Text-CNN | 84,18 | 83,42 | 77,89 | 64,74 |
| LSTM | 82,97 | 83,35 | 77,24 | 66,58 |
| GRU | 83,50 | 82,84 | 77,67 | 66,51 |
| PhoBERT | 90,01 | 88,93 | **86,89** | **72,17** |
| BERT4News | 86,39 | 86,20 | 86,16 | 62,62 |

- Bài gốc **không có baseline ML cổ điển** (không có TF-IDF + LR/SVM). Đây là khoảng trống trực tiếp cho đề tài.
- Con số "88,93" hay bị trích nhầm là F1 T1; thực ra là accuracy T2.
- Phân bố nhãn trong Hình 1 của bài khớp từng con số với CSV mirror `visolex/ViSpamReviews`, nên mirror là split v1 gốc.

**Dinh, Luu 2024 (split v2 chia lại 8:2, 14.284 / 1.589 / 3.963, KHÔNG so trực tiếp với v1), Bảng 6, bản gốc không metadata → tốt nhất có metadata:**

| Mô hình | Macro F1 T1 | Macro F1 T2 |
|---|---:|---:|
| TextCNN | 77,18 | 60,64 |
| PhoBERT | 85,58 → 87,22 (+mô tả SPhoBERT) | 71,56 → 73,49 |

**ViSoBERT (split v1, Bảng 2 của bài ghi ViSpamReviews 4 lớp, tức chỉ chạy T2):** ViSoBERT Acc 90,99, weighted F1 90,92, macro F1 **79,06**; PhoBERT-large trong cùng bảng: macro F1 76,88. Đây là mức trần hiện biết cho T2. Bài không báo T1.

**Kết quả của nhóm VeriReview (AI4Life, split v1 đã bỏ trùng 14.181 / 1.587 / 3.971), T1:** TF-IDF ký tự 2-5 + LR: 83,5; PhoBERT-base-v2: 89,5.

## 3. Pipeline kế thừa của Al-Zoubi, Mora, Faris (IEEE Access 2023, đã đọc toàn văn)

Số trang theo bản PDF trong thư mục này. Phần dưới do Claude đọc lại từ văn bản gốc (bản Gemini báo "xong" nhưng chưa ghi gì).

- **Mã hóa lời giải (trang 11):** vectơ D + 2 số thực trong [0, 1]. Hai phần tử đầu là C và gamma của SVM, đổi thang bằng min-max sang [0, 35000] và [0, 32] (công thức 10). D phần tử còn lại là **trọng số của từng đặc trưng**, nhân vào cột đặc trưng tương ứng. Vậy "weighted SVM" ở đây là trọng số đặc trưng liên tục, không phải chọn tập con nhị phân; D bằng số chiều đặc trưng của bộ dữ liệu (Bảng 4).
- **Hàm fitness (công thức 11, trang 11):** trung bình độ chính xác (accuracy) của SVM trên các fold. **Đơn mục tiêu**, không có phạt số đặc trưng hay thời gian.
- **Giao thức (trang 11-12):** CV 3 fold bên trong khi huấn luyện, và CV 10 fold bên ngoài (chạy lặp 10 lần, mỗi lần một phần dữ liệu để kiểm tra). Công thức 11 định nghĩa N là số mẫu của *tập kiểm tra*; nếu fitness thật sự tính trên fold kiểm tra thì chọn mô hình và báo kết quả dùng chung một dữ liệu, độ chính xác báo cáo sẽ lạc quan. Đây là cách đọc văn bản, cần xác nhận khi tái lập.
- **Dữ liệu:** tweet về sản phẩm bằng tiếng Ả Rập, Anh, Tây Ban Nha và tập đa ngôn ngữ; mỗi tập có các phiên bản biểu diễn TF-IDF, NGram-3, WE100/WE400 (embedding từ BERT, FastText, MUSE) và Words (one-hot).
- **Thí nghiệm:** (I) bộ phân loại truyền thống (J48, k-NN, NB...); (II) SVM kết hợp MVO, GA, PSO, GOA, SSA, HHO; (III) WSVM-HHO so với HHO-FsSVM (chọn đặc trưng); (IV) thêm đặc trưng cảm xúc. Chạy bằng MATLAB 2016.
- **Kết quả chính (accuracy, Bảng 11 và phần kết luận):** Ả Rập 89,565; Anh 88,163; Tây Ban Nha 71,913; đa ngôn ngữ 84,270 (độ lệch chuẩn 1,429 ở một con số trích trong bài).
- **Điểm yếu phương pháp luận (đối chiếu từ khóa Wilcoxon, Friedman, t-test, p-value trong văn bản: không có):**
  - không có kiểm định thống kê giữa các thuật toán, chỉ có độ lệch chuẩn;
  - chỉ báo accuracy, không báo macro F1 dù dữ liệu mất cân bằng;
  - chỉ một mục tiêu, nên không có đánh đổi độ chính xác và chi phí;
  - ngân sách đánh giá của các metaheuristic chỉ nằm ở Bảng 6 (dạng ảnh, văn bản trích ra không đọc được); cần xem trực tiếp trang 14 để biết có so cùng ngân sách hay không.

**Điểm thay thế của đề tài:** giữ ý tưởng "tối ưu đồng thời siêu tham số và trọng số/chọn đặc trưng", đổi sang **đa mục tiêu** (macro F1 và chi phí), dùng macro F1, nhiều seed, kiểm định thống kê và so cùng ngân sách; áp dụng cho tiếng Việt (chưa thấy công bố nào trên ViSpamReviews).

## 4. Khuôn của Anees et al. (preprint)

Làm sạch → 4 khối đặc trưng (TF-IDF từ 1-4 gram 10.000; n-gram ký tự 3-6 gram 1.500; CountVectorizer 1-2 gram 2.000; 43 đặc trưng ngôn ngữ học) → HHO (30 cá thể, 50 vòng) chọn 1.368/13.539 đặc trưng → SMOTEENN → stacking (ExtraTrees, RF, XGBoost, SVM; LR làm meta). Bài không nêu hàm fitness, không so với phương pháp chọn đặc trưng khác: đây là điểm yếu mà đề tài sửa (so cùng ngân sách, nhiều seed).