import matplotlib.pyplot as plt
from src.dataset import load_neu_dataset, CLASSES

DATA_PATH = "D:/Tài Liệu/Đồ án XLA/Surface Defect Projeect/Dataset/NEU-DET/train"  

imgs, lbls = load_neu_dataset(DATA_PATH)

plt.figure(figsize=(15, 6))
for idx, class_name in enumerate(CLASSES):
    # Tìm ảnh đầu tiên thuộc lớp này
    match_idx = (lbls == idx).nonzero()[0][0]
    
    plt.subplot(2, 3, idx + 1)
    plt.imshow(imgs[match_idx], cmap='gray')
    plt.title(f"{class_name.upper()}", fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig("sample_defects.png", dpi=300)
print("[+] Đã lưu ảnh mẫu 6 lớp khuyết tật vào file 'sample_defects.png'.")
plt.show()