# Huong Dan Runtime Advisor

Tai lieu nay giai thich cong cu `run_app.py --advisor-only`: no lam gi, doc ket qua ra sao, va dung cong cu nay nhu the nao de chon runtime mode phu hop truoc khi mo camera that.

## 1. Runtime Advisor La Gi

`run_app.py --advisor-only` la che do:

- khong mo webcam,
- khong chay detection realtime,
- khong mo giao dien camera,
- chi phan tich he thong va dua ra goi y runtime.

Lenh:

```powershell
python run_app.py --advisor-only
```

## 2. Muc Tieu Cua Cong Cu

Cong cu nay giup tra loi 4 cau hoi:

1. may hien tai dang manh den dau,
2. co GPU / CUDA / torch san sang hay khong,
3. nen uu tien model nao,
4. nen bat dau voi `high`, `medium`, `low`, hay `auto`.

## 3. Noi Dung Dau Ra Thuong Gap

Runtime advisor thuong hien:

- CPU, RAM, GPU, VRAM
- torch version
- CUDA build
- danh sach model local dang co
- mode runtime du kien
- `imgsz`, `max_det`, `device`, model de xuat

## 4. Y Nghia Tung Mode

### `high`

Phu hop khi:

- may co GPU tot,
- uu tien do chinh xac,
- co the chap nhan tai cao hon.

### `medium`

Phu hop khi:

- muon can bang FPS va do chinh xac,
- muon bat dau voi lua chon an toan,
- day la mode hay hop voi da so may dev.

### `low`

Phu hop khi:

- may yeu,
- dang chay CPU,
- webcam / he thong khong on dinh,
- can uu tien toc do va do ben.

### `auto`

Khong phai mode co dinh, ma la co che chon mode theo may dang dung. Thuong advisor se dua ra de xuat tren co so:

- co CUDA hay khong,
- VRAM nhieu hay it,
- model nao san co trong `models/`.

## 5. Cach Doc Ket Qua

Vi du neu advisor bao:

```text
medium: model=yolo11s.pt, device=cuda:0, imgsz=512, max_det=120
```

Ban co the hieu:

- mode khoi dong hop ly la `medium`,
- model uu tien la `yolo11s.pt`,
- se chay bang GPU `cuda:0`,
- kich thuoc anh input 512,
- gioi han detection moi frame la 120.

## 6. Cach Dung Cung Cac Lenh Khac

Quy trinh khuyen nghi:

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode medium
```

Neu da co model custom:

```powershell
python run_app.py --advisor-only
python run_app.py --model models/trained/best.pt --mode medium
```

## 7. Lien Quan Toi Cac Module Khac

Runtime advisor phu thuoc nhieu vao:

- `core/hardware_info.py`
- `core/runtime_advisor.py`
- `app/camera_runtime/bootstrap.py`
- `config/settings.yaml`

Neu goi y khong hop ly, day la nhom file nen debug dau tien.

## 8. Khi Nao Nen Chay Runtime Advisor

Nen chay trong cac truong hop:

- may moi vua cai repo,
- vua thay GPU / driver / torch,
- vua doi model local,
- camera realtime dang lag,
- muon so sanh giua `medium` va `low`,
- truoc khi demo tren may la.

## 9. Van De Thuong Gap

### Advisor bao CUDA nhung runtime van cham

Ly do co the la:

- model qua nang,
- `imgsz` qua cao,
- GPU dang bi app khac chiem,
- webcam output qua lon.

Thu:

```powershell
python run_app.py --mode low
python run_app.py --mode medium
```

### Advisor thay model local nhung ket qua nhan dien kem

Can tach ro:

- advisor chi goi y mode/runtime,
- khong danh gia chat luong nghiep vu cua model.

Neu muon model tot hon, can xem tiep:

- `training_guide.md`
- `models/trained/best.pt`
- dataset object detection thuc te.

## 10. Muc Tieu Khi Su Dung Dung Cach

Runtime advisor giup team:

- mo camera voi cau hinh hop ly hon,
- giam bug do chon sai mode,
- de huong dan may moi,
- de debug van de nang / lag / FPS thap mot cach co he thong.
