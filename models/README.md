# Model suy luận

File model không được commit (`models/pretrained/**`). Model ung thư cũ đã được xóa khỏi repo; máy tác giả còn 3 file detector cục bộ để test (vú, đại trực tràng) và chúng không đi kèm khi clone. Các checkpoint chung còn lại (`brain_classifier.pt`, `modality_classifier.pt`, `yolo11*.pt`) không phải 11 model chẩn đoán ung thư.

## Tình trạng hiện tại

`models/model_catalog.json` đang để trống có chủ đích: chưa tìm được 11 checkpoint miễn phí vừa có bằng chứng chất lượng đủ tin cậy, vừa khớp hợp đồng/runtime 2D hiện tại. `python scripts/fetch_model.py --all` vì vậy không tải gì.

Ứng viên CT 3D PanCancerSeg cho thận, gan, tụy và phổi có checkpoint công khai, nhưng mỗi tệp khoảng 816–820 MB, chạy bằng nnU-Net v2/CUDA; chúng không dùng được như YOLO 2D của ứng dụng. Baseline cùng challenge báo Dice tổn thương trung bình 0,328 trên 50 ca validation, nên chưa đủ căn cứ gọi là “chất lượng tốt”. R-Super có checkpoint thận/tụy, license CC BY-NC 4.0 và cũng cần pipeline CT 3D riêng. Máy phát triển hiện có RTX 3050 Ti 4 GB với PyTorch CUDA, nhưng chưa có cohort gắn nhãn phù hợp để đánh giá độc lập.

## Quy ước runtime

Thư mục đích: `models/pretrained/cancers/<key>/<modality>/`

| Loại | Tên file | Hợp đồng |
|---|---|---|
| Classifier | `classifier.pt` | Dict (không phải state_dict trần) gồm **bắt buộc** `model_state_dict`, `class_labels`, `num_classes`; tuỳ chọn `backbone` (resnet18/34/50, convnext_tiny, efficientnet_b0, `timm:*`), `head` (`mlp` hoặc `linear`), `image_size` (mặc định 320), `temperature`, `interpolation` |
| Detector | `detector.pt` hoặc `detector.onnx` | YOLO detect/segment, predict được ở 320×320 |
| Volume 3D | `volume.pt2` và `volume.json` | `torch.export` và manifest (`volume.pt` TorchScript cũ vẫn được hỗ trợ) |

Xuất model ở chế độ inference (`eval`) rồi lưu bằng định dạng `torch.export`:

```python
model.eval()
example = torch.randn(1, channels, depth, height, width)
exported = torch.export.export(model, (example,))
torch.export.save(exported, "volume.pt2")
```

Ghi lại phiên bản PyTorch dùng để export và dùng cùng phiên bản khi chạy inference; định dạng `.pt2` chưa đảm bảo tương thích giữa các phiên bản PyTorch.

Ví dụ lưu checkpoint classifier sau khi tự huấn luyện (thiếu `num_classes` sẽ bị từ chối khi probe):

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "class_labels": ["no_tumor", "glioma", "meningioma", "pituitary"],  # đúng thứ tự output
    "num_classes": 4,
    "backbone": "resnet18",      # phải khớp kiến trúc đã train
    "head": "mlp",
    "image_size": 320,           # đúng độ phân giải lúc train
    "temperature": 1.0,
}, "models/pretrained/cancers/brain/mri/classifier.pt")
```

Checkpoint ngoài (ResNet/timm-efficientnet) có thể chuyển bằng `python scripts/import_foreign_model.py --source-dir <thư mục model> --target <nhóm> --modality <modality>`.

Lưu ý: kết quả phân tích hiện **luôn** ở mức `uncertain` — model chỉ cung cấp nhãn và xác suất, ứng dụng không tự kết luận ác tính (xem `medical/pipeline.py::_classify_findings`).

Model mới chỉ được thêm vào catalog sau khi xác nhận license, tải đủ/hash, probe runtime và đánh giá trên test set đúng modality, tách theo bệnh nhân. Xem [báo cáo đánh giá](../docs/model_evaluation_report.md).
