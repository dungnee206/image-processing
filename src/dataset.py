import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

CLASSES = [
    'crazing',          # 0: Nứt rạn
    'inclusion',        # 1: Tạp chất
    'patches',          # 2: Mảng bám
    'pitted_surface',   # 3: Rỗ bề mặt
    'rolled-in_scale',  # 4: Vảy cán
    'scratches'         # 5: Trầy xước
]

def cv2_imread_utf8(img_path):
    """
    Đọc ảnh an toàn cho đường dẫn chứa tiếng Việt có dấu và khoảng trắng trên Windows.
    Đọc trực tiếp luồng byte nhị phân rồi giải mã qua OpenCV.
    """
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

def load_neu_dataset_recursive(base_dir, img_size=(200, 200)):
    """
    Duyệt đệ quy qua toàn bộ cây thư mục, nhận diện nhãn tự động và load ảnh xám.
    """
    images = []
    labels = []
    
    print(f"[*] Bắt đầu quét dữ liệu từ: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"[!] Lỗi: Không tìm thấy đường dẫn: {base_dir}")
        return np.array([]), np.array([])
        
    class_counts = {c: 0 for c in CLASSES}
    valid_extensions = ('.bmp', '.png', '.jpg', '.jpeg')
    
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                file_lower = file_name.lower().replace('-', '_')
                folder_lower = os.path.basename(root).lower().replace('-', '_')
                
                assigned_label = None
                
                for label_idx, class_name in enumerate(CLASSES):
                    clean_class_name = class_name.replace('-', '_')
                    if clean_class_name in folder_lower or file_lower.startswith(clean_class_name):
                        assigned_label = label_idx
                        break
                        
                if assigned_label is not None:
                    img_path = os.path.join(root, file_name)
                    img = cv2_imread_utf8(img_path)
                    
                    if img is not None:
                        if img.shape != img_size:
                            img = cv2.resize(img, img_size)
                        images.append(img)
                        labels.append(assigned_label)
                        class_counts[CLASSES[assigned_label]] += 1

    print("[*] Thống kê số lượng ảnh từng lớp nạp thành công:")
    for c, count in class_counts.items():
        print(f" -> Lớp '{c}': {count} ảnh")
        
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"[+] Đã nạp thành công: {len(images)} ảnh vào bộ nhớ, kích thước mảng: {images.shape}.")
    return images, labels

def get_train_test_split(images, labels, test_size=0.2, random_state=42):
    """
    Phân chia Stratified Train/Test để giữ đều tỉ lệ 6 lớp khuyết tật.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    print(f"[+] Phân chia dữ liệu: Train = {len(X_train)} ảnh | Test = {len(X_test)} ảnh.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # ĐƯỜNG DẪN CẬP NHẬT MỚI
    DATA_PATH = r"D:\Tài Liệu\DIP_Project\Surface_Defect_Project\Dataset\NEU-DET\train"
    
    imgs, lbls = load_neu_dataset_recursive(DATA_PATH)
    if len(imgs) > 0:
        X_tr, X_te, y_tr, y_te = get_train_test_split(imgs, lbls)
    else:
        print("[!] Không nạp được ảnh nào. Vui lòng kiểm tra lại đường dẫn!")