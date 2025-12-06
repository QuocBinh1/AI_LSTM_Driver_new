import csv
from pathlib import Path
import os

# 1. Xác định đường dẫn thư mục 'data'
# Đảm bảo script này nằm ở thư mục gốc của dự án (cùng cấp với 'data')
data_dir = Path("data") 

# 2. Kiểm tra xem thư mục có tồn tại không
if not data_dir.exists():
    print(f"Lỗi: Không tìm thấy thư mục '{data_dir.resolve()}'")
    print("Hãy đảm bảo bạn chạy script này từ thư mục gốc của dự án.")
else:
    stats = {}
    total_files = 0
    total_frames = 0

    # 3. Xác định các thư mục cha (ví dụ: 'eyes', 'mouth')
    categories = ["eyes", "mouth"]

    for category in categories:
        category_path = data_dir / category
        if not category_path.exists():
            print(f"Cảnh báo: Không tìm thấy thư mục con '{category}'")
            continue

        # 4. Lặp qua từng thư mục lớp (ví dụ: 'eyes_natural', 'eyes_sleepy')
        for class_dir in category_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                file_count = 0
                frame_count = 0

                # 5. Đếm file .csv và đếm dòng
                for csv_file in class_dir.glob("*.csv"):
                    file_count += 1
                    try:
                        with open(csv_file, mode='r', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            # Đếm tổng số dòng
                            num_rows = sum(1 for row in reader)
                            # Trừ 1 cho header, đảm bảo không bị âm
                            frame_count += max(0, num_rows - 1)
                    except Exception as e:
                        print(f"Lỗi khi đọc file {csv_file}: {e}")
                
                # 6. Lưu kết quả nếu có file
                if file_count > 0:
                    stats[class_name] = {"files": file_count, "frames": frame_count}
                    total_files += file_count
                    total_frames += frame_count

    # 7. In kết quả ra dạng bảng
    print("--- Thống kê dữ liệu thô ---")
    print(f"{'Lớp dữ liệu':<15} | {'Số file (.csv)':<15} | {'Tổng số frame':<15}")
    print("-" * 47)
    
    # Sắp xếp kết quả theo tên lớp cho đẹp
    for class_name in sorted(stats.keys()):
        data = stats[class_name]
        print(f"{class_name:<15} | {data['files']:<15} | {data['frames']:<15}")
        
    print("-" * 47)
    print(f"{'Tổng cộng':<15} | {total_files:<15} | {total_frames:<15}")