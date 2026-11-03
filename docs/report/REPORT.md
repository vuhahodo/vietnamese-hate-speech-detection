---
permalink: nlp-commerce-doctor/docs/report/report
title: REPORT
type: note
---

# Phát hiện review spam tiếng Việt bằng đặc trưng cổ điển và tối ưu tiến hóa đa mục tiêu

Báo cáo môn NLP. Phạm vi: nhiệm vụ T1 (spam / không spam) trên ViSpamReviews. Mọi con số truy được về `results/` (đường dẫn ghi cạnh số). Nhiệm vụ T2 (4 loại spam) chưa chạy, xem mục 7.

## 1. Đặt vấn đề

ViSpamReviews (Dinh, Luu, Nguyen 2022) có 19.870 review tiếng Việt. Bài gốc chỉ báo các mô hình mạng nơ-ron (PhoBERT macro F1 T1 = 86,89 trên split gốc) và **không có baseline cổ điển**. Câu hỏi:

- RQ1: pipeline cổ điển tốt nhất đạt bao nhiêu so với PhoBERT trên cùng split?
- RQ2: tối ưu tiến hóa đa mục tiêu có hơn HHO, GA, random, grid **cùng ngân sách đánh giá** không?
- RQ3: giảm được bao nhiêu đặc trưng mà macro F1 không giảm có ý nghĩa thống kê?

Pipeline kế thừa: Al-Zoubi, Mora, Faris (IEEE Access 2023), HHO đơn mục tiêu tối ưu đồng thời siêu tham số SVM và trọng số đặc trưng. Phần thay thế của đề tài: tối ưu **đa mục tiêu** (macro F1 và số đặc trưng), so cùng ngân sách, nhiều seed, có kiểm định.

## 2. Dữ liệu và giao thức

- Split gốc 14.306 / 1.590 / 3.974; sau khi bỏ trùng nguyên văn: **14.181 / 1.587 / 3.971** (`results/data_audit.json`). Cùng split và băm SHA-256 với dự án VeriReview.
- Fitness của mọi thuật toán: macro F1 trung bình CV 5 fold phân tầng **chỉ trên train**; chi² chọn đặc trưng được tính riêng trong từng fold.
- Dev dùng để chọn cấu hình sau tìm kiếm. Test chấm cho 3 cấu hình cố định trước khi xem test (`scripts/final_eval.py`). Test đã được đọc 6 lần trong toàn dự án, đều với cấu hình cố định (tái lập baseline 2 lần, độ nhạy gần trùng 2 lần, đánh giá cuối 1 lần, ViSoBERT 1 lần).
- Giới hạn: TF-IDF fit trên toàn train (không có nhãn nên không rò nhãn, nhưng IDF thấy cả fold kiểm tra).

## 3. Phương pháp

**Bốn khối đặc trưng** (`vispam/features.py`): B1 TF-IDF từ 1-2 gram (tách từ pyvi, 36.082 cột); B2 TF-IDF ký tự 2-5 gram (113.258); B3 36 thống kê bề mặt; B4 80 đếm từ điển quảng cáo/thương hiệu (log-scale, xem mục 7).

**Không gian tìm kiếm** (10 gen, `vispam/search/space.py`): tỉ lệ đặc trưng giữ lại của mỗi khối (xếp hạng chi², dưới 0,15 thì tắt khối), trọng số 4 khối (2^-2 đến 2^2, tương tự trọng số đặc trưng của Al-Zoubi nhưng ở mức khối), log10 C, loại bộ phân loại (LinearSVC hay LR cân bằng lớp). Mọi vectơ đều giải mã ra cấu hình hợp lệ.

**Hai mục tiêu:** tối đa macro F1 CV; tối thiểu số đặc trưng (dùng log10 khi tính hypervolume).

**Thuật toán** (`vispam/search/algorithms.py`), cùng 500 lần đánh giá, quần thể 30, 10 seed:
- đề xuất: NSGA-II (pymoo), MODE (DE/rand/1/bin + sắp xếp không trội + crowding);
- đối chứng: HHO (đơn mục tiêu, như bài kế thừa), GA đơn mục tiêu, GA tổng có trọng số (có phạt số đặc trưng), random, grid.

Mọi thuật toán, đơn hay đa mục tiêu, được chấm bằng **tập không trội của mọi điểm nó đã đánh giá** và hypervolume với điểm tham chiếu (F1 = 0,5; log10 = 5,3) chốt trước khi chạy.

## 4. Kết quả tìm kiếm (T1)

Nguồn: `results/search_summary_t1.csv`, `results/search_stats_t1.json`; hình `docs/report/figures/pareto_t1.png`, `hypervolume_convergence_t1.png`.

| Thuật toán | Hypervolume TB (sd) | Hạng TB | F1 CV tốt nhất | F1 dev cấu hình tốt nhất | Đặc trưng ở điểm knee |
|---|---:|---:|---:|---:|---:|
| MODE | 1,376 (0,072) | 1,4 | 0,8574 | 0,8450 | ~20.000 |
| NSGA-II | 1,325 (0,077) | 2,2 | 0,8570 | 0,8437 | ~22.400 |
| HHO | 1,272 (0,051) | 3,2 | 0,8556 | 0,8442 | ~50.300 |
| Random | 1,234 (0,062) | 4,0 | 0,8561 | 0,8442 | ~28.000 |
| Grid | 1,236 (0,021) | 4,3 | 0,8552 | 0,8420 | ~28.300 |
| GA + phạt | 1,037 (0,189) | 5,9 | 0,8574 | 0,8424 | ~14.900 |
| GA (chỉ F1) | 0,778 (0,259) | 7,0 | 0,8596 | 0,8433 | ~34.100 |

- Friedman trên hypervolume: thống kê 50,0, p = 4,7e-9.
- Wilcoxon theo cặp, hiệu chỉnh Holm (12 so sánh): MODE hơn GA, GA+phạt, grid và random (p Holm = 0,023); NSGA-II hơn GA và GA+phạt (0,023) nhưng **không** có ý nghĩa so với grid (0,059), random (0,068), HHO (0,111). MODE và NSGA-II không khác nhau (0,551). Với 10 seed, p nhỏ nhất có thể đạt là 0,002 nên hiệu chỉnh Holm nhiều so sánh khó đạt ngưỡng 0,05 cho các chênh lệch vừa phải.
- **Độ chính xác không khác nhau giữa các thuật toán:** F1 CV tốt nhất 0,855 đến 0,860; F1 dev 0,842 đến 0,845. Cấu hình chọn tay dùng cả 4 khối đạt dev 0,8455, ngang cấu hình tiến hóa. Ưu thế của tối ưu đa mục tiêu nằm ở **chi phí**, không ở độ chính xác.
- Thận trọng: hypervolume thiên về thuật toán đa mục tiêu vì chúng chủ động đẩy về vùng ít đặc trưng. So sánh công bằng nhất là với GA+phạt, và MODE/NSGA-II vẫn hơn. F1 CV cao hơn F1 dev khoảng 0,013 ở mọi thuật toán (chưa điều tra nguyên nhân).

## 5. Đánh giá cuối trên test (T1, 3.971 review)

Nguồn: `results/final_test_t1.json`, `results/final_test_t1_phobert.json`, `results/latency_t1.json`. Ba cấu hình cố định theo dev: BASE (LR + B2, baseline VeriReview), FULL (LinearSVC C=0,1, cả 4 khối), EA (điểm knee của MODE seed 4, F1 dev cao nhất trong 10 seed).

| Mô hình | Số đặc trưng | Macro F1 [CI 95%] | Độ trễ 1 review (trung vị) | Kích thước |
|---|---:|---:|---:|---:|
| PhoBERT-base-v2 (AI4Life) | không áp dụng | **0,8945** [0,884; 0,905] | ~62 ms (theo lô, CPU) | lớn |
| ViSoBERT (Kaggle GPU, 3 epoch, văn bản thô) | không áp dụng | 0,8835 [0,872; 0,894] | ~71 ms (theo lô, CPU) | lớn |
| EA | 24.060 | 0,8630 [0,851; 0,874] | 2,7 ms | 8,7 MB |
| FULL | 149.456 | 0,8619 [0,850; 0,874] | 5,2 ms | 10,7 MB |
| BASE | 113.258 | 0,8355 [0,822; 0,849] | 2,9 ms | 7,8 MB |

Kiểm định theo cặp trên cùng mẫu (chênh macro F1, bootstrap 95%, McNemar):
- EA - BASE = +2,75 điểm [1,7; 3,8], McNemar p < 0,001.
- FULL - BASE = +2,64 điểm [1,6; 3,6], p < 0,001.
- EA - FULL = +0,11 điểm [-0,75; 0,89], p = 1,0 (không phân biệt được).
- ViSoBERT - EA = +2,05 điểm [0,9; 3,1], p < 0,001; PhoBERT - ViSoBERT = +1,11 điểm [0,09; 2,14], McNemar p = 0,032 (khác biệt biên giới).
- PhoBERT - EA = +3,15 điểm [2,0; 4,3], p < 0,001; PhoBERT - BASE = +5,9 điểm [4,6; 7,2], p < 0,001.

Độ trễ đầu-cuối (chuỗi văn bản đến dự đoán, một review mỗi lần, CPU) đo bằng `scripts/measure_latency.py`; pipeline đầu-cuối khớp 100% dự đoán dev của mã tìm kiếm. Độ trễ PhoBERT là suy luận theo lô 32, không cùng điều kiện với mô hình cổ điển.

## 6. Thảo luận

- **RQ1.** Pipeline cổ điển 4 khối đạt 0,863 trên test, hơn baseline TF-IDF ký tự (0,8355) 2,7 điểm, nhưng vẫn kém PhoBERT 3,2 điểm có ý nghĩa thống kê. Con số 86,89 của bài gốc dùng split chưa bỏ trùng và không so trực tiếp được; số PhoBERT so sánh được là 0,8945 trên cùng split của ta.
- **RQ2.** Đa mục tiêu (MODE, NSGA-II) hơn GA đơn mục tiêu rõ ràng và, riêng MODE, hơn random và grid về hypervolume. NSGA-II chưa tách khỏi random/grid sau hiệu chỉnh Holm. Độ chính xác không khác biệt.
- **RQ3.** Điểm knee dùng 24.060 / 149.456 đặc trưng (16%) với macro F1 test không khác FULL (chênh +0,11 điểm, p = 1,0). Độ trễ giảm khoảng một nửa so với FULL (5,2 so với 2,7 ms); phần lớn thời gian còn lại là tách từ và trích đặc trưng nên giảm số đặc trưng thêm cũng không nhanh hơn nhiều.
- Chưa trả lời: khối nào giải thích loại spam nào (RQ4) cần T2.

## 7. Hạn chế và việc chưa làm

- Chỉ T1. T2 chưa chạy (khoảng 10 giờ CPU), SPAM-1 chỉ có 61 dòng test nên F1 T2 sẽ nhiễu.
- Ngân sách 500 lần đánh giá, quần thể 30 (kế hoạch ban đầu 2.000), do thời gian chạy.
- B4 phải được log-scale: giá trị thô lên tới 128 làm LinearSVC chậm 7 đến 25 lần. Baseline ở `results/baselines_*.csv` dùng B4 chưa chuẩn hóa, nên không so trực tiếp với bảng ở mục 5 (dev 0,851 của LinearSVC trong CSV là cấu hình khác).
- Kết quả stacking T2 trong `results/baselines_t2.csv` bất thường (dev 0,614 so với 0,735 mô hình đơn), chưa điều tra; không dùng để kết luận.
- Gần trùng train-test: 71/3.971 dòng test có cosine >= 0,90 với train, bỏ đi thì macro F1 của baseline giảm 0,0065 (`results/near_duplicate_sensitivity.json`); khoảng tin cậy chồng nhau nên không kết luận rò rỉ.
- Dữ liệu không có shop, thời gian, người đánh giá; nhãn spam khác khung Y0/Y1/YU của VeriReview (Y0/Y1/YU là nhãn ưu tiên kiểm tra, không phải nhãn thật/giả). Độ đồng thuận nhãn gốc κ = 0,72.
- Một cấu hình EA (seed 4) được chọn theo dev trong 10 seed: có thể lạc quan nhẹ nhưng test là tập độc lập.

## 8. Tái lập

```
python scripts/run_baselines.py --task t1          # baseline cổ điển
python scripts/run_search.py --task t1 --budget 500 --seeds 0-9 --workers 8   # ~10 giờ
python scripts/summarize_search.py --task t1       # hypervolume, Wilcoxon, Friedman
python scripts/final_eval.py                       # chấm test một lần
python scripts/phobert_predict_test.py --model <thư mục phobert_spam_model>
python scripts/compare_phobert.py
python scripts/measure_latency.py                  # độ trễ đầu-cuối + xuất mô hình
python scripts/make_figures.py
```

Dữ liệu `data/raw/ViSpamReviews.csv` (CC-BY-NC) không nằm trong repo; mã `vispam/data.py` kiểm SHA-256.