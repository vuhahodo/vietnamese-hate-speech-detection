---
permalink: nlp-commerce-doctor/docs/agents/gemini-tasks-a
title: GEMINI_TASKS_A
type: note
---

# Giao việc cho Gemini, đợt A: đọc bài, giao thức đánh giá, baseline cổ điển

Bạn là agent thực thi (worker) cho đề tài trong repo này. Claude điều phối và duyệt kết quả của bạn. Đọc `docs/PLAN.md` (toàn bộ) và `docs/papers/README.md` trước khi làm. Làm hết các việc dưới đây theo thứ tự, không cần hỏi lại; gặp chỗ mơ hồ thì chọn phương án hợp lý nhất, ghi quyết định vào báo cáo.

## Luật cứng (vi phạm là kết quả bị loại)

1. **Không chạy lệnh git ghi** (`add`, `commit`, `push`, `reset`, `checkout`, `tag`...). Chỉ được `git status`, `git diff`, `git log`. Claude sẽ commit.
2. **Không bịa số liệu.** Mọi con số trong báo cáo và tài liệu phải lấy từ lần chạy thật (kèm lệnh và file kết quả) hoặc chép từ PDF (kèm trang/bảng). Không biết thì ghi "không xác định".
3. **Test chỉ chấm một lần.** Tập test chỉ được dùng trong đúng hai trường hợp: (a) cấu hình tái lập LR + khối B2 (xem T4), (b) phân tích gần trùng T5. Mọi lựa chọn mô hình/siêu tham số dùng CV trên train và dev. Mọi script phải mặc định không đụng test, test chỉ chạy khi có cờ `--final-test`.
4. Mã, tên biến, comment, commit message bằng **tiếng Anh**; **không emoji**. Tài liệu `.md` bằng tiếng Việt có dấu.
5. Không đưa dữ liệu hoặc PDF vào chỗ git theo dõi; `data/`, `cache/`, `docs/papers/*.pdf|txt` đã nằm trong `.gitignore`. Không sửa `.gitignore`.
6. Không sửa `vispam/data.py` và `tests/test_data.py` (nếu thấy lỗi thì ghi vào báo cáo). Không tick `[x]` trong `docs/PLAN.md`; chỉ đổi `[ ]` thành `[~]` và ghi dòng bằng chứng.
7. Không dùng GPU, không gọi API trả phí. Máy có 12 luồng CPU; dùng tối đa `n_jobs=10`. Python 3.10; các thư viện trong `requirements.txt` đã có (`pymoo` chưa cài, không cần ở đợt này). Cần thư viện khác (lightgbm, xgboost...) thì `pip install` rồi thêm vào `requirements.txt`.
8. Chạy lệnh Python với `PYTHONIOENCODING=utf-8`. Mỗi việc xong phải chạy `python -m pytest -q tests` và pass trước khi sang việc kế.

## T1. Đọc 2 bài Al-Zoubi (P0.1)

File: `docs/papers/A_Multilingual_Spam_Reviews_Detection.pdf` (IEEE Access 2023, WSVM-HHO). Bài thứ hai là Al-Zoubi, Mora, Faris, Qaddoura, *A hybrid TwinSVM-HHO model for multilingual spam review detection...* (Expert Systems with Applications 287, 2025, doi 10.1016/j.eswa.2025.128160): tìm file PDF của nó trong `docs/papers/`, `D:\Change_C_to_D\Downloads`, `C:\Users\Admin\Downloads` (tên có thể chứa TwinSVM, HHO, spam, eswa, S0957417425017804). Nếu không thấy thì ghi vào báo cáo và bỏ qua phần bài thứ hai.

Trích văn bản bằng `pypdf` ra `docs/papers/<tên>.txt`, rồi cập nhật `docs/papers/README.md` thêm mục mới "Pipeline của Al-Zoubi et al. (đã đọc toàn văn)" gồm, mỗi ý kèm số trang/bảng:
- cách mã hóa lời giải HHO (độ dài vectơ, phần nào là siêu tham số, phần nào là chọn đặc trưng/trọng số);
- **hàm fitness** (công thức đúng như bài);
- tham số HHO (số cá thể, số vòng lặp), số lần chạy lặp, cách chia dữ liệu và có dùng CV hay không;
- các thuật toán metaheuristic so sánh và có so cùng ngân sách đánh giá hay không;
- bảng kết quả chính (chép nguyên số), có kiểm định thống kê không;
- những điểm yếu phương pháp luận bạn thấy (ví dụ thiếu seed, không cùng ngân sách).
Sửa lại cột "Trạng thái" của hai bài trong bảng mục 1 thành "đã tải, đã đọc" nếu đúng.

## T2. Giao thức đánh giá (P0.3)

Tạo `vispam/evaluate.py` và `tests/test_evaluate.py`:
- `macro_f1`, `per_class_f1`, `confusion` (dựa trên scikit-learn);
- `bootstrap_ci(y_true, y_pred, metric, n=1000, seed=0)` trả (điểm, thấp, cao) khoảng 95% percentile;
- `paired_bootstrap_diff` cho hiệu macro F1 giữa hai mô hình trên cùng mẫu;
- `mcnemar(y_true, pred_a, pred_b)` trả thống kê và p-value (dùng phép kiểm chính xác nhị thức khi số ô bất đồng < 25, ngược lại chi bình phương có hiệu chỉnh liên tục);
- `latency_ms(predict_fn, texts, repeats=3)` đo thời gian trung bình mỗi review trên CPU, bỏ lần chạy đầu để làm nóng.
Test đơn vị trên dữ liệu tổng hợp (ví dụ MCNemar với bảng đã biết kết quả; bootstrap CI chứa điểm ước lượng).

## T3. Khối đặc trưng (P1.1)

Tạo `vispam/features.py` và `tests/test_features.py`. Mọi khối: `fit(train_texts)` chỉ trên train, `transform(texts)` trả `scipy.sparse.csr_matrix`, có `names` (tên đặc trưng) và cache ma trận ra `cache/` (khóa theo hash cấu hình + hash văn bản) để các lần chạy sau nhanh.
- **B1** TF-IDF từ 1-2 gram trên văn bản đã tách từ bằng `pyvi.ViTokenizer`, `min_df=2`, `sublinear_tf=True`, tối đa 50.000 đặc trưng.
- **B2** TF-IDF ký tự 2-5 gram (`analyzer="char_wb"`), `min_df=2`, `sublinear_tf=True`, tối đa 100.000 đặc trưng. Đây là cấu hình của baseline đã đạt macro F1 test 0,835 ở dự án VeriReview; nếu cần chỉnh thì ghi lý do.
- **B3** đặc trưng thống kê bề mặt (khoảng 30-45 đặc trưng, chuẩn hóa bằng `MaxAbsScaler` fit trên train): số ký tự/từ, tỉ lệ chữ hoa, tỉ lệ chữ số, số emoji, số dấu câu và tỉ lệ, độ dài chuỗi lặp ký tự dài nhất, tỉ lệ từ lặp, có URL, có số điện thoại, tỉ lệ ký tự không phải tiếng Việt/Latin, tỉ lệ từ không dấu, số từ tiếng Anh, v.v.
- **B4** từ điển: danh sách viết tay các cụm quảng cáo/khuyến mãi, từ chỉ thương hiệu/liên hệ ("shop", "inbox", "zalo", "fanpage", "giảm giá", "mua ngay", tên thương hiệu phổ biến...) và cụm khen chung chung. Ghi rõ trong docstring: danh sách này viết tay từ mô tả loại spam trong bài 2207.14636 mục 3 và Bảng 1; không lấy từ tập test. Trả về số lần xuất hiện theo nhóm.
`transform` phải tạo cùng số cột cho mọi đầu vào, không rò rỉ thông tin của dev/test vào `fit`.

## T4. Baseline cổ điển (P1.2)

Tạo `scripts/run_baselines.py` (tham số: `--task {t1,t2}`, `--final-test`) và `configs/baselines.yaml` nếu cần. Với **cả hai nhiệm vụ** (T1 nhãn `label`, T2 nhãn `spam_type`):
1. Bộ phân loại: `LogisticRegression` (cân bằng lớp), `LinearSVC` (cân bằng lớp), `ComplementNB`, `SGDClassifier(loss="modified_huber")`, và một mô hình cây (LightGBM hoặc `RandomForest`) chạy trên B3+B4 (đặc trưng dày, ít chiều).
2. Tổ hợp khối đặc trưng: từng khối B1, B2, B3+B4; B1+B2; B1+B2+B3+B4 (ghép ngang bằng `scipy.sparse.hstack`).
3. Lưới siêu tham số nhỏ cho mỗi mô hình (ví dụ C trong {0,1; 1; 10}, alpha NB trong {0,1; 0,5; 1}); chọn bằng CV phân tầng 5 fold trên train theo macro F1. Báo: macro F1 CV (trung bình ± độ lệch), macro F1 dev, F1 từng lớp trên dev, độ trễ.
4. **Stacking:** lấy 3 mô hình tốt nhất theo CV, kết hợp bằng `StackingClassifier` với meta-learner LR, đánh giá như trên.
5. Kết quả vào `results/baselines_t1.csv` và `results/baselines_t2.csv` (một dòng mỗi tổ hợp khối x mô hình, có cột seed và cấu hình).
6. **Tái lập (chỉ lần này được chấm test):** với cờ `--final-test`, chấm **một lần** cấu hình LR cân bằng lớp + chỉ khối B2 (siêu tham số mặc định C=1) trên test T1, in macro F1 + bootstrap CI + F1 lớp spam, và ghi `results/repro_t1_test.json`. ĐKHT: macro F1 trong khoảng 0,835 ± 0,005 (số của dự án VeriReview). Nếu lệch, tìm nguyên nhân (đừng chỉnh cho khớp), ghi vào báo cáo.

## T5. Độ nhạy gần trùng (P1.3)

Tạo `scripts/near_duplicate_sensitivity.py`. Dùng ma trận TF-IDF ký tự B2 fit trên train; với mỗi dòng test tính cosine lớn nhất với các dòng train (dùng phép nhân ma trận thưa theo khối để không tốn bộ nhớ). Báo: số dòng test có cosine >= 0,90 (kỳ vọng khoảng 71/3.971 theo dự án VeriReview; ghi số thực), phân bố cosine theo phân vị, macro F1 của mô hình T4.6 trên toàn bộ test và trên phần test sau khi bỏ các dòng gần trùng (dùng lại dự đoán, không chạy lại chọn mô hình), kèm bootstrap CI. Xuất `results/near_duplicate_sensitivity.json`. Đây là lần thứ hai được dùng test, và chỉ để phân tích độ nhạy.

## T6. Báo cáo cuối đợt

Viết `docs/agents/REPORT_A.md` (tiếng Việt, ngắn gọn, không văn hoa):
- Việc nào xong (T1-T5), việc nào dở, vì sao.
- Bảng kết quả chính lấy từ các file `results/` (dán số thật), kèm lệnh đã chạy.
- Số test đã dùng bao nhiêu lần và ở đâu.
- Thời gian chạy từng script (để Claude tính ngân sách cho đợt tối ưu).
- Mọi bất thường: lệch so với kỳ vọng, lỗi đã gặp, quyết định tự đưa ra, câu hỏi mở.
- Danh sách file đã tạo/sửa (`git status`).
Cuối cùng chạy `python -m pytest -q tests` và dán dòng kết quả.
## Kênh trao đổi với Claude (bắt buộc)

Thư mục `docs/agents/bus/` là kênh hai chiều (xem `README.md` ở đó). Trước khi bắt đầu, và sau khi xong **mỗi** việc T1-T6:
1. Thêm một dòng vào `docs/agents/bus/gemini_status.md` theo mẫu trong README của bus.
2. Đọc `docs/agents/bus/claude_to_gemini.md`; nếu có chỉ dẫn mới thì làm theo (không trái luật cứng ở trên).
3. Có câu hỏi cần quyết thì ghi vào `docs/agents/bus/gemini_questions.md` và làm tiếp việc không phụ thuộc.
