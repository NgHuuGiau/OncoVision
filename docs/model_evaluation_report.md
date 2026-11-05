# Đánh giá 11 nhóm model ung thư

Ngày chạy: 2026-10-20. Đây là đánh giá kỹ thuật trên model và dữ liệu công khai; không phải kiểm định lâm sàng. Các checkpoint ung thư được đánh giá trong bảng lịch sử bên dưới đã bị xóa sau đó theo yêu cầu người dùng. Kết quả của chúng không còn đại diện cho model hiện cài.

## Trạng thái sau khi xóa model

Đã xóa toàn bộ `models/pretrained/cancers/` trong repo. Giữ lại các checkpoint nền/chung: `brain_classifier.pt`, `modality_classifier.pt` và `yolo11*.pt`. App hiện báo **1/11** nhóm có model runtime do vẫn có `brain_classifier.pt`; 10 nhóm còn lại đang chờ model. Trên máy tác giả còn 3 file detector đặt cục bộ để test (bị `.gitignore` chặn, clone mới không có) — xem mục "Test lại 2026-10-27" bên dưới.

## Test lại 2026-10-27 (model cục bộ, dữ liệu công khai)

Chỉ có 2 model khác nhau (file `colorectal/endoscopy/detector.pt` trùng sha256 với `colonoscopy/detector.pt`), 8 nhóm còn lại chưa có file nào nên không test được.

| Model | Dữ liệu test | mAP@0.5 | mAP@0.5:0.95 | Kết luận |
|---|---|---|---|---|
| `breast/ultrasound/detector.pt` | 300 ảnh BUSI (120 benign, 120 malignant, 60 normal), bbox lấy từ mask | 0,91 | 0,75 | Khoanh đúng tổn thương (IoU 0,73–0,98 trên mẫu kiểm tra tay), nhãn benign/malignant khớp. Lớp `normal` không bao giờ được dự đoán; 1/10 ảnh normal bị báo nhầm malignant 0,40. mAP cao có thể do model đã train trên BUSI → chưa chứng minh tổng quát hoá. |
| `colorectal/colonoscopy/detector.pt` | 612 ảnh CVC-ClinicDB, bbox lấy từ mask polyp | 0,00007 | 0,00002 | **Không định vị được**: 123/127 dự đoán phủ >80% diện tích ảnh (bbox gần bằng cả khung). Đây là model phân loại ảnh được đóng gói thành YOLO detect, không dùng được để khoanh vùng polyp. |

Lệnh tái lập: `python run_medical.py evaluate-detector --model <file> --data <thư mục images/labels YOLO>`.

Tìm được checkpoint mới PanCancerSeg cho thận, gan, tụy và phổi (Apache-2.0), mỗi checkpoint khoảng 816–820 MB, nhưng là nnU-Net 3D CT chứ không tương thích với detector 2D hiện tại. Baseline cùng challenge báo Dice tổn thương trung bình **0,328** trên 50 ca validation-public; đây không phải điểm của bốn checkpoint chuyên biệt, nhưng không đủ bằng chứng để gọi bộ đó là “chất lượng tốt”. R-Super cho thận/tụy có checkpoint công khai 607 MB nhưng dùng license CC BY-NC 4.0 và cũng cần pipeline 3D riêng. Máy hiện không có CUDA; còn 2.4 GB RAM khả dụng. Vì vậy chưa tải chúng vào runtime hoặc tuyên bố đã test chất lượng. Nguồn: [PanCancerSeg weights](https://huggingface.co/KS987/PanCancerSeg-Specialized-weights), [baseline và kết quả validation](https://github.com/Kappapapa123/CVPR2026-PanCancerSeg-baseline), [R-Super weights](https://huggingface.co/AbdomenAtlas/R-SuperPancreasKidney), [R-Super paper/repo](https://github.com/MrGiovanni/R-Super).

Hiện chưa tìm được đủ 11 checkpoint miễn phí có license rõ, test độc lập phù hợp và chạy được trên runtime hiện tại. Catalog để trống để `--all` không tải lại các model cũ. Các kết quả dưới đây chỉ là snapshot lịch sử trước khi xóa:

## Sàng lọc nguồn thay thế theo 11 nhóm

| Nhóm | Ứng viên đã tìm | Kết quả sàng lọc |
|---|---|---|
| Não | [BraTS 2021 winner, KAIST](https://github.com/rixez/Brats21_KAIST_MRI_Lab) | Trọng số public, segmentation MRI đa chuỗi 3D; cần môi trường nnU-Net/CUDA, không dùng được như classifier 2D hiện tại. |
| Vú | [CLC-YOLO](https://github.com/mfazrinizar/CLC-YOLO) | Ultrasound segmentation; báo cáo Dice 89.21% trên BUS-BRA, nhưng tác giả ghi rõ chưa có external/clinical validation và trọng số tải được từ release chưa xác nhận. |
| Dạ dày | [GutCore](https://huggingface.co/SehunFromDaegu/GutCore) | Đúng tác vụ ung thư dạ dày nội soi, nhưng checkpoint yêu cầu đăng nhập/chấp nhận điều kiện và license PolyForm Noncommercial; không phải tải tự do không điều kiện. |
| Đại trực tràng | [DepthPolyp](https://github.com/ReaganWu/DepthPolyp) | Có checkpoint phát hiện polyp và test trên bộ colonoscopy; polyp không đồng nghĩa ung thư đại trực tràng nên không tính là model ung thư đúng đích. |
| Tiền liệt tuyến | [PI-CAI algorithms](https://pi-cai.grand-challenge.org/algorithms/) | Đúng bpMRI và có benchmark lớn; thuật toán được cấp quyền tự động theo yêu cầu trên Grand Challenge, không phải tải checkpoint trực tiếp không điều kiện. |
| Cổ tử cung | [Smart-CCS](https://github.com/hjiangaz/Smart-CCS) | Có trọng số pretrained và test workflow, nhưng cho cell/WSI screening, không phải ảnh soi cổ tử cung mà catalog app yêu cầu. |
| Thận | [PanCancerSeg](https://huggingface.co/KS987/PanCancerSeg-Specialized-weights) / [RenalCLIP](https://github.com/dt-yuhui/RenalCLIP) | Có model CT 3D và model biểu diễn ung thư thận; đều cần pipeline riêng. PanCancerSeg specialist chưa có score độc lập xác minh trong model card. |
| Tụy | [R-Super](https://huggingface.co/AbdomenAtlas/R-SuperPancreasKidney) | Checkpoint CT 3D, 6.9K ca u thận/tụy trong huấn luyện; license CC BY-NC 4.0, không tương thích runtime 2D. |
| Tuyến giáp | [Thyroid nodule classifier](https://huggingface.co/ljubomir/thyroid-nodule-classifier) | Checkpoint benign/malignant cho ROI siêu âm; cần ROI cắt sẵn, chưa xác minh test ngoài độc lập phù hợp. |
| Gan | [PanCancerSeg](https://huggingface.co/KS987/PanCancerSeg-Specialized-weights) | Có checkpoint CT 3D chuyên gan; chưa có score độc lập xác minh cho checkpoint này, không khớp runtime 2D. |
| Phổi | [MONAI LUNA16 detection](https://huggingface.co/MONAI/lung_nodule_ct_detection) | Model CT 3D phát hiện nốt phổi, không kết luận ung thư; nốt phổi không đồng nghĩa ung thư. |

Kết quả: chỉ ghi nhận các ứng viên cần thẩm định, chưa cài ứng viên nào. Không có đủ 11 model vừa khớp đúng bệnh và modality, vừa có checkpoint/license truy cập được, vừa có test độc lập đủ tin cậy. Các model ghi “candidate” không được tính là model đã test.

## Kết quả đã đo

| Nhóm | Artifact/tác vụ hiện có | Kết quả thực tế | Nhận xét |
|---|---|---|---|
| Não | YOLO MRI, 4 nhãn u/không u | Nạp và predict được trên ảnh tổng hợp; chưa có chỉ số trên ảnh MRI gán nhãn | Chưa chứng minh khả năng phân loại bệnh nhân hoặc định vị u trên cohort độc lập. |
| Vú | ONNX vú tổng quát; YOLO mammogram; YOLO siêu âm | Model siêu âm: tác giả công bố test mAP@0.5 **0,737**, mAP@0.5:0.95 **0,506**, malignant recall **0,645**. Hai model còn lại chưa có kiểm tra ngoài bộ huấn luyện | Model siêu âm được huấn luyện trên BUSI; split ảnh cùng một trung tâm, không phải external validation. Recall ác tính thấp. [Model card và protocol](https://huggingface.co/SevdanurGenc/Lesion-Detection-in-Breast-Ultrasound-Images-with-YOLOv8) |
| Dạ dày | Chưa có model trong registry | Không có model để chạy | Tìm thấy model CT phân giai đoạn T1–T4, nhưng khác tác vụ phát hiện sớm và không có cohort CT công khai tương ứng; chưa đưa vào hệ thống. [Repo](https://github.com/18846068128/AI_CT_GastricCancer_TStaging2025) |
| Đại trực tràng | YOLO nội soi, lớp polyp/other | CVC-ClinicDB, **384 ảnh có cặp mask**, CPU, imgsz 320: precision **0,870**, recall **0,820**, mAP@0.5 **0,797**, mAP@0.5:0.95 **0,508** | Đánh giá ngoài Kvasir. Mask nhị phân được đổi thành một hộp bao mỗi ảnh; mirror có 384 ảnh khớp trong bộ 612 ảnh. Đây là đo phát hiện polyp, không phải chẩn đoán ung thư. [Dataset mirror](https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb). Báo cáo máy: `tmp-ds/cvc-clinicdb/evaluation.json`. |
| Tuyến tiền liệt | Classifier patch mô bệnh học H&E, 3 lớp | Nạp được; chưa có chỉ số trên cohort ngoài | Không phải model MRI tuyến tiền liệt; chỉ phân loại patch, không định vị hoặc chẩn đoán toàn lam kính. |
| Cổ tử cung | Classifier tế bào Pap, 5 lớp | Nạp được; chưa có chỉ số ngoài | Không phải model soi cổ tử cung/định vị tổn thương. Chưa có bộ test độc lập khớp chính xác 5 lớp. |
| Thận | YOLO CT 2D, tumor/cyst/stone | Nạp và predict được trên ảnh tổng hợp; chưa đo trên CT gán nhãn | Bộ KiTS23 là CT 3D segmentation, còn pipeline hiện tại không có kiểm định 3D tương ứng. [KiTS23](https://kits-challenge.org/kits23/) |
| Tụy | Chưa có model trong registry | Không có model để chạy | PanDx là ứng viên mã nguồn mở cho PDAC trên CT 3D, nhưng không cùng hợp đồng/runtime của app và cần môi trường GPU; chưa import hay benchmark. [Repo](https://github.com/han-liu/PanDx) |
| Tuyến giáp | Classifier benign/malignant trên ROI siêu âm | Nạp được; chưa có external score | Không có detector/segmenter đi kèm để tạo ROI. Bộ UD-TN công khai chỉ cho sample 6 ảnh ở repo, quá nhỏ để kết luận chất lượng. [Dataset sample](https://github.com/18811755633/Sample-of-UD-TN) |
| Gan | YOLO CT 2D, tumor | Nạp và predict được trên ảnh tổng hợp; chưa đo trên CT gán nhãn | LiTS có nhãn CT 3D, nhưng cần pipeline chuyển thể tích và chuẩn hoá CT được xác minh trước khi tính mAP. [LiTS](https://pmc.ncbi.nlm.nih.gov/articles/PMC10631490/) |
| Phổi | YOLO X-quang, 14 dấu hiệu ngực | Nạp và predict được trên ảnh tổng hợp; chưa đo trên bộ X-quang gán nhãn | Đây là model dấu hiệu lồng ngực, không phải model ung thư phổi đặc hiệu. VinDr-CXR yêu cầu tài khoản, DUA và đào tạo CITI nên không tải khi chưa có quyền truy cập. [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) |

## Phạm vi chạy

- Hệ thống trước đó: smoke test pass **1/1**; unit tests pass **383/383**. Không sửa code trong lượt đánh giá này.
- Artifact hiện có ở chín nhóm ung thư; các artifact đều nạp và predict được trên ảnh tổng hợp. Registry báo runtime 8/11; dạ dày và tụy chưa có artifact, còn tuyến tiền liệt chỉ có classifier H&E nên chưa đáp ứng MRI được quảng bá.
- Tải CVC-ClinicDB mirror vào `tmp-ds/cvc-clinicdb/`; đã xác nhận 384 ảnh và 384 mask có cặp. Bản mirror thiếu phần ảnh còn lại nên không gọi đây là toàn bộ dataset.
- Bằng chứng runtime chỉ xác nhận tương thích kỹ thuật. Nó không đo độ nhạy trên bệnh nhân, không thay cho đánh giá theo patient-level split hoặc kiểm định đa trung tâm.

## Việc cần làm tiếp

1. Bổ sung external test có nhãn đúng modality cho từng model 2D, giữ nguyên phân chia theo bệnh nhân và ghi hash/version nguồn dữ liệu.
2. Chọn riêng runtime 3D cho CT/MRI volume; không tính điểm cho mô hình 2D bằng cách coi các lát cắt của cùng một bệnh nhân là mẫu độc lập.
3. Chỉ thêm model dạ dày/tụy sau khi xác nhận license, weights, preprocessing, checkpoint compatibility và cohort test phù hợp. Hai repo tìm được không thể thay cho model phát hiện ung thư dùng chung trong app.
