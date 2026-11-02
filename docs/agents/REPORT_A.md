---
permalink: nlp-commerce-doctor/docs/agents/report-a-1
title: REPORT_A
type: note
---

# Báo cáo đợt A (bản Claude kiểm chứng)

Ngày 02/11/2026. Mọi số dưới đây đọc trực tiếp từ `results/`. Bản Gemini tự viết được giữ ở `REPORT_A_gemini_UNVERIFIED.md` kèm các sai lệch ghi ở mục 5; không dùng làm nguồn số liệu.

## 1. Trạng thái

| Việc | Trạng thái | Bằng chứng |
|---|---|---|
| T1 đọc bài Al-Zoubi 2023 | Xong (Claude làm lại; bản Gemini báo xong nhưng chưa ghi gì) | `docs/papers/README.md` mục 3. Bài ESWA 2025 chưa có PDF |
| T2 giao thức đánh giá | Xong | `vispam/evaluate.py`, `tests/test_evaluate.py` (8 test) |
| T3 khối đặc trưng B1-B4 | Xong, B2 đã sửa cấu hình | `vispam/features.py`, `tests/test_features.py` (7 test); tổng 19 test pass |
| T4 baseline T1 và T2 | Xong | `results/baselines_t1.csv`, `results/baselines_t2.csv` (22 dòng mỗi file) |
| T4.6 tái lập 0,835 | Đạt | `results/repro_t1_test.json`: 0.8345 (CI [0.8224, 0.8469]) |
| T5 gần trùng | Xong | `results/near_duplicate_sensitivity.json` |

## 2. Lỗi cấu hình B2 và cách xử lý

Bản giao việc của Claude ghi sai B2 (`char_wb`, `min_df=2`, 100.000 đặc trưng). Baseline VeriReview dùng `analyzer="char"`, `min_df=3`, 150.000 đặc trưng. Lần tái lập đầu cho 0,825, ngoài ĐKHT; Gemini giải thích là do bỏ trùng, nhưng split giống hệt nên giải thích đó sai. Claude xác nhận nguyên nhân trên tập dev (không dùng test): cấu hình đúng cho dev macro F1 0,8233, khớp báo cáo AI4Life; cấu hình sai cho 0,7999. B2 đã sửa, kết quả cũ chuyển vào `results/superseded/` làm bằng chứng.

## 3. Kết quả

### T1 (spam / không spam), xếp theo dev macro F1
| Khối | Mô hình | CV macro F1 (5 fold) | Dev macro F1 |
|---|---|---:|---:|
| b1+b2+b3+b4 | LinearSVC | 0.8536 ± 0.0040 | 0.8507 |
| b1+b2+b3+b4 | LogisticRegression | 0.8510 ± 0.0056 | 0.8453 |
| b1+b2+b3+b4 | SGDClassifier | 0.8436 ± 0.0046 | 0.8381 |
| stacking(b1+b2+b3+b4) | StackingClassifier | 0.8384 ± 0.0081 | 0.8381 |
| b1+b2 | LogisticRegression | 0.8392 ± 0.0098 | 0.8372 |
| b1+b2 | SGDClassifier | 0.8384 ± 0.0101 | 0.8317 |

- Tái lập baseline VeriReview (LR cân bằng + B2, C=1): **test macro F1 0.8345**, CI 95% [0.8224, 0.8469], F1 spam 0.7659 (mốc 0,835).
- Thêm B1 và các khối B3/B4 nâng dev macro F1 của LR từ 0.8217 (chỉ B2) lên 0.8453 (B1+B2+B3+B4). Đây là CV/dev, chưa kiểm trên test.
- Stacking top 3 không vượt mô hình đơn tốt nhất: CV 0.8384, dev 0.8381.

### T2 (4 loại spam), xếp theo dev macro F1
| Khối | Mô hình | CV macro F1 (5 fold) | Dev macro F1 |
|---|---|---:|---:|
| b1+b2+b3+b4 | LogisticRegression | 0.7130 ± 0.0250 | 0.7347 |
| b1+b2+b3+b4 | LinearSVC | 0.7058 ± 0.0249 | 0.7323 |
| b1+b2 | LogisticRegression | 0.7058 ± 0.0251 | 0.7321 |
| b2 | LogisticRegression | 0.7023 ± 0.0158 | 0.7207 |
| b1+b2 | LinearSVC | 0.7014 ± 0.0229 | 0.7197 |
| b1+b2+b3+b4 | SGDClassifier | 0.6770 ± 0.0300 | 0.7114 |

- Bài gốc (Dinh et al., split gốc, **test**): PhoBERT 72,17; ViSoBERT 79,06 (Nguyen et al. 2023). Số dev của ta **không so trực tiếp** được với các số test đó; so sánh chính thức chỉ làm ở bước chấm test cuối (P2.5). SPAM-1 chỉ có 61 dòng test nên macro F1 T2 dao động mạnh.
- Stacking T2 kém bất thường: CV 0.6083, dev 0.6136, thấp hơn mọi mô hình đơn tốt. Chưa điều tra; giả thuyết (chưa kiểm): bộ meta-learner dùng `class_weight="balanced"` làm tăng báo nhầm SPAM-1, hoặc các estimator nhận chung ma trận gộp. Không dùng stacking T2 làm kết luận.

### T5 (gần trùng)
- 71/3971 dòng test có cosine >= 0,90 với train (khớp AI4Life).
- Macro F1 toàn test 0.8345 (CI [0.8224, 0.8469]); sau khi bỏ dòng gần trùng 0.828 (CI [0.8147, 0.8426]); chênh -0.0065. Hai khoảng tin cậy chồng nhau nhiều, nên chưa đủ để nói rò rỉ gần trùng thổi phồng kết quả.

## 4. Giới hạn cần ghi nhớ

- **Độ trễ trong CSV (`latency_ms_per_item`) chỉ đo `predict` trên ma trận đặc trưng tính sẵn**, không gồm bước biến văn bản thành đặc trưng (TF-IDF, tách từ pyvi). Nó thấp hơn thực tế nhiều; độ trễ đầu-cuối phải đo lại trước khi báo cáo hay viết model card.
- Có cảnh báo hội tụ của LinearSVC (35 lần ở T1, 45 lần ở T2). Cấu hình LinearSVC đã được Gemini đổi (`tol`, `max_iter`); số của LinearSVC chưa phải kết quả đã hội tụ hoàn toàn.
- CV chọn siêu tham số rồi báo lại chính điểm CV đó nên hơi lạc quan; dev mới là số dùng để so.

## 5. Số lần chạm tập test và độ tin cậy của Gemini

Tập test đã được dùng trong 4 lần chạy, đều với cấu hình cố định, không dùng để chọn mô hình hay siêu tham số: (1) tái lập với B2 sai; (2) tái lập với B2 đúng; (3) T5; (4) T5 chạy lại bởi Gemini, kết quả trùng. Bản Gemini ghi "đúng 2 lần" là sai.

Các sai lệch trong báo cáo Gemini đã phát hiện: số test của `test_evaluate` (báo 12, thực tế 8) và `test_features` (báo 19, thực tế 7); báo T1 xong khi chưa ghi gì vào README; bảng kết quả lấy từ lần chạy cũ sai B2 nhưng ghép với số repro mới; mô tả B2 còn là `char_wb`; so dev của ta với test của bài gốc để kết luận "ngang ngửa PhoBERT". Quy tắc làm việc: mọi báo cáo của Gemini phải được đối chiếu với file trong `results/` trước khi dùng.

## 6. Thời gian chạy (cho ngân sách đợt tối ưu)

Từ lần chạy của Claude trên laptop 12 luồng, `--n-jobs 5` mỗi job, hai job chạy song song: T1 (22 cấu hình CV 5 fold + stacking + tái lập test) khoảng 16 phút; T2 khoảng 44 phút (kể cả chờ tranh CPU với T1 và T5).

Thời gian của một lần đánh giá fitness (CV 5 fold trên ma trận đã cache) chưa được đo riêng; cần đo ở bước P2.1 trước khi chốt ngân sách.
