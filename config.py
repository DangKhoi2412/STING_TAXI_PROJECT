"""
config.py - File cấu hình trung tâm cho toàn bộ dự án STING.

Mọi tham số (đường dẫn dữ liệu, bounding box, lưới, ngưỡng truy vấn...)
đều được khai báo tại đây để tránh hard-code rải rác trong các module.
"""

# ==============================================================================
# 1. CẤU HÌNH DỮ LIỆU
# ==============================================================================

# Đường dẫn tới file dữ liệu NYC Taxi
DATA_PATH = "data/taxi_data.csv"

# Số dòng tối đa được load khi đọc dữ liệu (dùng để test cho nhẹ máy).
# Đặt None nếu muốn load toàn bộ dataset.
SAMPLE_ROWS = None

# Danh sách các cột cần sử dụng từ dataset gốc
COLUMNS = [
    "tpep_pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "payment_type",
]

# Loại thanh toán bằng thẻ tín dụng (credit card).
# Chỉ thanh toán bằng thẻ mới ghi nhận tip chính xác;
# thanh toán tiền mặt ghi tip = 0.0 → làm sai lệch phân tích.
PAYMENT_TYPE_CREDIT = 1

# ==============================================================================
# 2. BOUNDING BOX - GIỚI HẠN KHÔNG GIAN CỦA THÀNH PHỐ NEW YORK
# ==============================================================================

# Vĩ độ (Latitude) hợp lệ cho khu vực NYC
LAT_MIN = 40.5
LAT_MAX = 40.9

# Kinh độ (Longitude) hợp lệ cho khu vực NYC
LON_MIN = -74.25
LON_MAX = -73.7

# ==============================================================================
# 3. CẤU HÌNH LƯỚI PHÂN CẤP (HIERARCHICAL GRID)
# ==============================================================================

# Kích thước lưới tại tầng đáy (bottom layer).
# Ví dụ: grid_size = 64 → lưới đáy có 64 x 64 = 4096 ô.
GRID_SIZE = 64

# Số tầng phân cấp trong Quad-tree.
# Tầng 0 (root) chứa 1 ô → Tầng 1 chứa 4 ô → ... → Tầng cuối = bottom layer.
NUM_LAYERS = 4

# ==============================================================================
# 4. CẤU HÌNH TRUY VẤN STING (TOP-DOWN QUERY)
# ==============================================================================

# Biến mục tiêu để phân tích (cột tip_amount)
TARGET_VARIABLE = "tip_amount"

# Ngưỡng mật độ: ô phải chứa ít nhất n > DENSITY_THRESHOLD điểm dữ liệu
DENSITY_THRESHOLD = 10

# Ngưỡng trung bình: mean(tip_amount) của ô phải > MEAN_THRESHOLD
MEAN_THRESHOLD = 2.0

# ==============================================================================
# 5. CẤU HÌNH TRỰC QUAN HOÁ (VISUALIZATION)
# ==============================================================================

# Tên file output cho bản đồ Folium
FOLIUM_OUTPUT = "output/hotspot_map.html"

# Tên file output cho biểu đồ Matplotlib
MATPLOTLIB_OUTPUT = "output/hotspot_grid.png"
