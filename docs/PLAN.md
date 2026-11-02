---
permalink: nlp-commerce-doctor/docs/plan
title: PLAN
type: note
---

# Kế hoạch: Phát hiện review spam tiếng Việt bằng ML cổ điển và tối ưu tiến hóa đa mục tiêu

Đề tài môn NLP (thầy Đặng Đại Thọ), đồng thời là nhánh tín hiệu văn bản cho VeriReview (AI4Life). Làm xong ở repo này rồi mới chuyển phần cần thiết sang VeriReview (Giai đoạn 5).

Quy ước: `[ ]` chưa làm, `[~]` đang làm, `[x]` xong và có bằng chứng (đường dẫn file kết quả). Chỉ tick `[x]` khi đạt mọi ĐKHT (điều kiện hoàn thành).

## 0. Phát biểu đề tài

- **Bài toán:** ViSpamReviews, 2 nhiệm vụ: (T1) spam / không spam; (T2) loại spam: NO-SPAM, SPAM-1 review giả, SPAM-2 chỉ nói thương hiệu, SPAM-3 không liên quan.
- **Pipeline kế thừa:**
  - Chỗ dựa có phản biện: Al-Zoubi, Mora, Faris (IEEE Access 2023; ESWA 2025). Trong pipeline này, HHO tối ưu đồng thời siêu tham số SVM và trọng số/chọn đặc trưng cho phát hiện review spam.
  - Khuôn khối đặc trưng + stacking: Anees et al. 2025 (preprint).
  - Dữ liệu, split và baseline: Dinh, Luu, Nguyen 2022. Bài này **không có baseline ML cổ điển**: đây là khoảng trống của RQ1.
  - Danh mục bài và số liệu gốc: `docs/papers/README.md`.
- **Thành phần thay thế (đóng góp):** thay HHO đơn mục tiêu bằng tối ưu tiến hóa **đa mục tiêu** (NSGA-II và DE đa mục tiêu) chọn đồng thời khối đặc trưng, siêu tham số và trọng số ensemble, cho ra tập Pareto giữa macro F1 và chi phí (số đặc trưng hoặc độ trễ).
- **Câu hỏi nghiên cứu:**
  - RQ1: Pipeline cổ điển tốt nhất đạt bao nhiêu so với PhoBERT/ViSoBERT trên cùng split?
  - RQ2: Tối ưu đa mục tiêu có hơn HHO, GA, random search, grid search **cùng ngân sách đánh giá** không (hypervolume, macro F1 test, nhiều seed)?
  - RQ3: Giảm bao nhiêu đặc trưng/độ trễ mà macro F1 giảm không có ý nghĩa thống kê?
  - RQ4: Khối đặc trưng nào được chọn, và chúng giải thích được loại spam nào (dùng làm bằng chứng cho VeriReview)?

### Bài học từ thử nghiệm ViHOS trước (phải áp dụng)
1. EA chỉ dò 1 ngưỡng thì overfit và thua một ngưỡng tinh chỉnh đơn giản. Không gian tìm kiếm phải đủ lớn để EA có lý do tồn tại.
2. Fitness tính bằng cross-validation trên train, không tính trên một tập dev duy nhất.
3. Mọi phương pháp so cùng số lần đánh giá fitness; nhiều seed; báo cả kết quả âm.
4. Test chỉ chấm một lần, sau khi đã chốt cấu hình trên train/dev.

### Đã biết từ AI4Life (không làm lại, chỉ tái lập)
- Split nguồn sau khi bỏ trùng hệt: train 14.181, dev 1.587, test 3.971 (`AI4Life/docs/VISPAM_TRAINING_REPORT.md`).
- TF-IDF ký tự 2-5 + LR cân bằng: macro F1 test 0,835. PhoBERT-base-v2: 0,895.
- Rò rỉ gần trùng nhỏ: 71/3.971 dòng test có cosine ≥ 0,90 với train; bỏ đi thì F1 baseline 0,835 → 0,828. Chỉ là kiểm tra phụ, không phải đóng góp chính.

## 1. Cấu trúc repo

```
vispam/                 package Python
  data.py               tải, kiểm SHA-256, chuẩn hóa, bỏ trùng, split
  features.py           các khối đặc trưng (có cache ma trận)
  models.py             bộ phân loại, stacking
  evaluate.py           macro F1, bootstrap CI, McNemar, độ trễ
  search/               space.py (mã hóa cá thể), nsga2.py, mode.py, hho.py, ga.py, baselines.py
scripts/                điểm chạy: prepare_data, run_baselines, run_search, final_eval, make_figures
configs/                YAML cho từng thí nghiệm
results/                CSV/JSON/PNG nhỏ, có commit (bằng chứng)
docs/
  PLAN.md               file này
  papers/README.md      danh mục bài báo + số liệu gốc (PDF không commit)
  report/               báo cáo môn, slide
tests/                  pytest
data/                   KHÔNG commit (giấy phép CC-BY-NC, truy cập có kiểm soát)
```

## 2. Giai đoạn

### R. Dọn repo
- [x] **R1** Lưu bản cũ: tag `archive/vihsd-hate-speech` (commit `42cf912`) và bản sao toàn bộ thư mục ở `D:\HocTap\N4_K1\_archive_nlp_vihsd_vihos\`.
- [x] **R2** Xóa mã ViHSD/ViHOS cũ khỏi cây làm việc; viết lại README, `.gitignore`, `requirements.txt`.
- [~] **R3** Đã commit cục bộ `6200420`; chưa push (cần bạn cho phép). Commit "refactor: replace ViHSD project with ViSpamReviews research skeleton" và push (repo đang **public**). Cân nhắc đổi tên repo GitHub, ví dụ `vietnamese-spam-review-ea`. ĐKHT: `git ls-files` không còn file ViHSD; không có dữ liệu hoặc PDF trong git.

### P0. Tài liệu và dữ liệu
- [~] **P0.1 Bài báo.** Đã tải 5 bài arXiv và đọc bảng kết quả (`docs/papers/README.md`). Đã đọc toàn văn bài Al-Zoubi 2023 (mục 3 của README đó). Còn: bài ESWA 2025 chưa có PDF, xác nhận nơi đăng của 2207.14636 và 2402.12625, đọc Bảng 6 (dạng ảnh) của bài 2023 để biết quần thể và số vòng lặp của HHO. Tải về `docs/papers/` (không commit PDF):
  - Dinh, Luu, Nguyen. Detecting Spam Reviews on Vietnamese E-commerce Websites. arXiv 2207.14636.
  - Dinh, Luu. Metadata Integration for Spam Reviews Detection on Vietnamese E-commerce Websites. IJALP 2024, arXiv 2405.13292.
  - Anees et al. An Optimized ML Classifier for Detecting Fake Reviews Using Extracted Features. arXiv 2511.21716 (preprint).
  - Nguyen et al. ViSoBERT. arXiv 2310.11166 (mức trần PLM).
  - Một bài NSGA-II cho chọn đặc trưng đã qua phản biện (tìm thêm; ứng viên: Compact NSGA-II, arXiv 2402.12625).
  
  ĐKHT: `docs/papers/README.md` có thông tin xuất bản đã đối chiếu trang gốc, bảng kết quả gốc (T1, T2, macro F1) chép đúng từ bài, ghi rõ split và tiền xử lý của từng bài.
- [x] **P0.2 Dữ liệu.** Tải CSV mirror `visolex/ViSpamReviews` vào `data/raw/`. ĐKHT: SHA-256 khớp `675ba637…2690`; tái lập đúng 14.181 / 1.587 / 3.971; thống kê nhãn T2 theo split; `tests/test_data.py` pass. Bằng chứng: `results/data_audit.json`, `tests/test_data.py` (4 pass); hash khớp; phân bố nhãn khớp Hình 1 bài gốc. Lưu ý: test sau bỏ trùng còn 3.971 (bài gốc 3.974), nên khi so với bài gốc phải báo thêm số trên 3.974 dòng gốc. SPAM-1 chỉ có 61 dòng test: macro F1 T2 dao động lớn, bắt buộc có khoảng tin cậy.
- [~] **P0.3 Giao thức đánh giá.** `vispam/evaluate.py`: macro F1, F1 từng lớp, bootstrap 95% CI (1.000 lần), McNemar giữa 2 mô hình, độ trễ CPU trên 1.000 review. ĐKHT: có test đơn vị; quy tắc "test chấm 1 lần" ghi trong README. Bằng chứng: `vispam/evaluate.py`, `tests/test_evaluate.py` (8 test evaluate pass, tổng 19 test pass cả repo). Lưu ý: độ trễ trong CSV baseline chỉ đo `predict` trên ma trận tính sẵn, chưa gồm trích đặc trưng.

### P1. Tái lập và baseline (RQ1)
- [~] **P1.1 Khối đặc trưng:**
  - B1: TF-IDF từ 1-2 gram (tách từ pyvi);
  - B2: TF-IDF ký tự 2-5 gram;
  - B3: thống kê bề mặt: độ dài, tỉ lệ chữ hoa, emoji, ký tự lặp, số/URL/số điện thoại, dấu câu;
  - B4: từ điển: tên thương hiệu, cụm quảng cáo, từ khen chung chung (ghi nguồn từ điển);
  - B5 (tùy chọn): xác suất PhoBERT từ Kaggle, chỉ dùng làm điểm so sánh lai.
  
  ĐKHT: mỗi khối có cache, chỉ fit trên train; test đơn vị cho B3/B4. Bằng chứng: `vispam/features.py` (B1-B4, Composite, cache), `tests/test_features.py` (7 test features pass, tổng 19 pass).
- [~] **P1.2 Bộ phân loại:** LR, LinearSVC, ComplementNB, SGD, LightGBM hoặc RF trên B3/B4, và stacking. Chạy cả T1 và T2. ĐKHT: `results/baselines_{t1,t2}.csv`; LR + B2 tái lập 0,835 ± 0,005 trên T1; bảng so với số trong bài gốc. Bằng chứng: `scripts/run_baselines.py`, `results/baselines_t1.csv`, `results/baselines_t2.csv`, `results/repro_t1_test.json` (repro test macro F1 0.8345 [0.8224, 0.8469], delta -0.0005).
- [~] **P1.3 Độ nhạy gần trùng.** ĐKHT: tái lập con số 71/3.971 và F1 trên test đã lọc cho mô hình tốt nhất. Bằng chứng: `scripts/near_duplicate_sensitivity.py`, `results/near_duplicate_sensitivity.json` (khớp đúng 71/3.971 mẫu test, delta macro F1 -0.0065).

### P2. Tối ưu tiến hóa (RQ2, RQ3)
- [ ] **P2.1 Không gian tìm kiếm** (`vispam/search/space.py`):
  - mặt nạ nhị phân 4-5 khối đặc trưng;
  - mỗi khối: tỉ lệ đặc trưng giữ lại theo chi² (liên tục);
  - siêu tham số: C của LR/SVM, alpha của NB;
  - trọng số stacking (simplex).
  
  ĐKHT: giải mã mọi vectơ ra cấu hình hợp lệ; ước tính thời gian 1 lần đánh giá (CV 5 fold) dưới 5 giây trên laptop nhờ cache.
- [ ] **P2.2 Mục tiêu:** f1 = −macro F1 (CV 5 fold trên train), f2 = số đặc trưng (hoặc độ trễ đo được). Tùy chọn ràng buộc precision spam ≥ ngưỡng, xử lý bằng Deb's rules.
- [ ] **P2.3 Thuật toán:**
  - đề xuất: NSGA-II (pymoo) và DE đa mục tiêu tự cài (DE/rand/1/bin + non-dominated sorting);
  - đối chứng: HHO (kế thừa từ Anees), GA đơn mục tiêu, random search, grid search.
  
  ĐKHT: cùng ngân sách (ví dụ 2.000 lần đánh giá); 10 seed; log hội tụ.
- [ ] **P2.4 So sánh:** hypervolume (cùng điểm tham chiếu), macro F1 trên dev của điểm knee, Wilcoxon và Friedman giữa các thuật toán. ĐKHT: `results/search_summary.csv`, hình Pareto front, hình hội tụ; báo cả khi NSGA-II không hơn random search.
- [ ] **P2.5 Đánh giá cuối:** chọn 1-3 cấu hình từ Pareto front theo dev, chấm test 1 lần, bootstrap CI, McNemar với baseline P1 và PhoBERT. ĐKHT: `results/final_test.json`.

### P3. Phân tích (RQ4)
- [ ] **P3.1** Tần suất chọn từng khối đặc trưng trên các Pareto front; khối nào quan trọng cho từng loại spam ở T2.
- [ ] **P3.2** Giải thích từng review: top n-gram/đặc trưng đóng góp (hệ số tuyến tính) và loại spam dự đoán. ĐKHT: 20 ví dụ đúng/sai có phân tích lỗi.
- [ ] **P3.3** Hạn chế: không có metadata shop/thời gian, nhãn nguồn khác nhãn Y0/Y1 của VeriReview, κ = 0,72 của bộ dữ liệu.

### P4. Báo cáo môn
- [ ] **P4.1** Báo cáo: đặt vấn đề, liên quan, phương pháp (phát biểu bài toán đa mục tiêu, mã hóa, toán tử), thí nghiệm, kết quả, thảo luận, hạn chế. ĐKHT: mọi số trong báo cáo truy được về `results/`.
- [ ] **P4.2** Slide và demo nhỏ (script dự đoán 1 câu kèm giải thích).
- [ ] **P4.3** Lệnh chạy lại toàn bộ từ đầu ghi trong README; chạy thử trên máy sạch.

### P5. Chuyển sang VeriReview (sau khi P1-P3 xong)
- [ ] **P5.1** Trong AI4Life, tạo nhánh `task/A1-vispam-ea`; chép `vispam/` vào `research/vispam_ea/`, cùng `results/` và tóm tắt báo cáo. Không chép dữ liệu.
- [ ] **P5.2** Xuất mô hình được chọn + `model_card.json` (theo mục 4.4 của `KE_HOACH_CHIEN_LUOC_VERIREVIEW.md`): phiên bản, dữ liệu, chỉ số có CI, độ trễ, ngưỡng, giới hạn.
- [ ] **P5.3** Ánh xạ sang kế hoạch VeriReview: A0.3 (giao thức đánh giá), A1.1 (baseline đủ model card và độ trễ), A1.2 (so sánh có ý nghĩa thống kê với PhoBERT), A6.1 (báo cáo kỹ thuật). Loại spam T2 và đặc trưng chính trở thành dòng bằng chứng trên thẻ review.
- [ ] **P5.4** Mở PR, Codex review theo mục 7.3 của kế hoạch VeriReview.

## 3. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| EA không hơn random search | Vẫn là kết quả hợp lệ; nhấn mạnh RQ3 (giảm chi phí) và phân tích Pareto |
| Đánh giá chậm vì TF-IDF lớn | Cache ma trận theo khối; chọn theo chi² trên ma trận đã cache; chạy song song 12 luồng |
| Overfit vào CV | So điểm CV với dev; báo khoảng cách |
| Giấy phép dữ liệu | Không commit dữ liệu/PDF; repo public chỉ chứa mã và số liệu tổng hợp |
| Số liệu bài gốc không khớp giữa các nguồn (86,89 hay 88,93) | Lấy theo bảng trong PDF bản đã đăng, ghi phiên bản |

## 4. Thứ tự làm

R3 → P0.1 + P0.2 (song song) → P0.3 → P1.1 → P1.2 → P1.3 → P2.1 → P2.2 → P2.3 → P2.4 → P2.5 → P3 → P4 → P5.