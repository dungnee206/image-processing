import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from dataset import load_neu_dataset_recursive, get_train_test_split, CLASSES
from preprocessing import preprocess_dataset
from features import extract_features_dataset

def train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lưu bộ chuẩn hóa
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    
    # 2. Khởi tạo danh sách mô hình
    models = {
        "SVM": SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric="mlogloss", random_state=42)
    }
    
    best_acc = 0.0
    best_model_name = ""
    best_model = None
    
    print("\n" + "=" * 65)
    print("                 KẾT QUẢ ĐÁNH GIÁ CÁC MÔ HÌNH ML                ")
    print("=" * 65)
    
    for name, model in models.items():
        print(f"\n[*] Đang huấn luyện: {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f" -> {name} - Độ chính xác (Accuracy): {acc * 100:.2f}%")
        
        # In chi tiết Precision, Recall, F1
        print(f"\n[+] Báo cáo phân loại của {name}:")
        print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
        
        # Vẽ Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f"Confusion Matrix - {name} (Acc: {acc*100:.2f}%)", fontweight="bold")
        plt.xlabel("Dự đoán (Predicted)")
        plt.ylabel("Thực tế (Actual)")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"cm_{name.lower()}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        
        # Cập nhật mô hình tốt nhất
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
            
    # Lưu mô hình có kết quả cao nhất
    best_model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    
    print("=" * 65)
    print(f"[+] Mô hình tối ưu nhất: {best_model_name} với Accuracy = {best_acc * 100:.2f}%")
    print(f"[+] Đã lưu mô hình tốt nhất vào: {best_model_path}")
    print(f"[+] Đã lưu ma trận nhầm lẫn các mô hình vào thư mục: {output_dir}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    DATA_PATH = r"D:\Tài Liệu\DIP_Project\Surface_Defect_Project\Dataset\NEU-DET\train"
    
    # Bước 1: Nạp dữ liệu
    imgs, lbls = load_neu_dataset_recursive(DATA_PATH)
    
    if len(imgs) > 0:
        # Bước 2: Chia tập dữ liệu Train (80%) / Test (20%)
        X_train_raw, X_test_raw, y_train, y_test = get_train_test_split(imgs, lbls)
        
        # Bước 3: Tiền xử lý (Mặc định dùng CLAHE để làm nổi vân khuyết tật)
        print("\n[*] Áp dụng tiền xử lý CLAHE trên tập dữ liệu...")
        X_train_prep = preprocess_dataset(X_train_raw, method="clahe")
        X_test_prep = preprocess_dataset(X_test_raw, method="clahe")
        
        # Bước 4: Trích xuất đặc trưng
        print("\n[*] Trích xuất đặc trưng (Histogram + GLCM + LBP)...")
        X_train_feats = extract_features_dataset(X_train_prep)
        X_test_feats = extract_features_dataset(X_test_prep)
        
        # Bước 5: Huấn luyện và đánh giá
        train_and_evaluate_models(X_train_feats, y_train, X_test_feats, y_test)