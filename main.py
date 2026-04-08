"""
main.py - Orchestrator (Nhạc trưởng) cho toàn bộ pipeline STING.

Luồng xử lý end-to-end:
────────────────────────
    Step 1: Đọc dữ liệu (DataLoader)
    Step 2: Tiền xử lý (Preprocessor)
    Step 3: Xây dựng lưới phân cấp + nạp dữ liệu (HierarchicalGrid)
    Step 4: Bottom-Up aggregation (tự động trong feed_data)
    Step 5: Top-Down Query với Pruning (StingQuery)
    Step 6: BFS Clustering — 8-way adjacency (RegionFormation)
    Step 7: Trực quan hoá (Visualizer — Matplotlib + Folium)

Cách chạy:
    cd STING_Taxi_Project
    python main.py
"""

import config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.grid import HierarchicalGrid
from src.sting import StingQuery
from src.clustering import RegionFormation
from src.visualization import Visualizer
from src.utils import timer, print_header


def main() -> None:
    """
    Hàm chính — chạy toàn bộ pipeline STING từ đầu đến cuối.
    """
    print_header("STING: Statistical Information Grid — NYC Taxi Project")

    # ==================================================================
    # STEP 1: ĐỌC DỮ LIỆU
    # ==================================================================
    with timer("Step 1 — Đọc dữ liệu"):
        loader = DataLoader()
        df = loader.load()
        loader.check_missing_values()
        loader.summary()

    # ==================================================================
    # STEP 2: TIỀN XỬ LÝ
    # ==================================================================
    with timer("Step 2 — Tiền xử lý"):
        preprocessor = Preprocessor()
        df_clean = preprocessor.transform(df)

    # ==================================================================
    # STEP 3 & 4: XÂY DỰNG LƯỚI + BOTTOM-UP AGGREGATION
    # ==================================================================
    with timer("Step 3 & 4 — Xây dựng Grid + Bottom-Up Stats"):
        grid = HierarchicalGrid()
        grid.feed_data(df_clean)
        grid.print_summary()

    # ==================================================================
    # STEP 5: TOP-DOWN QUERY (PRUNING)
    # ==================================================================
    with timer("Step 5 — Top-Down Query"):
        # Ngưỡng nới lỏng hơn sau khi đã lọc chỉ credit card:
        #   min_n = 5: ít nhất 5 chuyến/ô (đủ tin cậy thống kê)
        #   min_mean = 1.0: trung bình tip ≥ $1.0
        query = StingQuery(
            grid=grid,
            min_n=5,
            min_mean=1.0,
        )
        relevant_cells = query.execute()

    # Kiểm tra: nếu không có ô relevant → dừng sớm
    if not relevant_cells:
        print("\n⚠ Không tìm thấy ô relevant nào. Hãy thử:")
        print("  - Giảm DENSITY_THRESHOLD trong config.py")
        print("  - Giảm MEAN_THRESHOLD trong config.py")
        print("  - Tăng SAMPLE_ROWS để load nhiều dữ liệu hơn")
        return

    # ==================================================================
    # STEP 6: BFS CLUSTERING (8-WAY ADJACENCY)
    # ==================================================================
    with timer("Step 6 — BFS Clustering"):
        region = RegionFormation(relevant_cells)
        clusters = region.form_clusters()

    # ==================================================================
    # STEP 7: TRỰC QUAN HOÁ
    # ==================================================================
    with timer("Step 7 — Trực quan hoá"):
        viz = Visualizer(grid=grid, clusters=clusters)

        # 7a. Biểu đồ Matplotlib (lưới 2D)
        print("\n[Main] Đang vẽ biểu đồ Matplotlib...")
        viz.plot_clusters_matplotlib(show=False)

        # 7b. Bản đồ Folium (bản đồ tương tác NYC)
        print("[Main] Đang tạo bản đồ Folium...")
        viz.plot_clusters_folium()

    # ==================================================================
    # KẾT QUẢ TỔNG HỢP
    # ==================================================================
    print_header("KẾT QUẢ TỔNG HỢP")
    print(f"  Dữ liệu gốc        : {len(df):,} dòng")
    print(f"  Sau tiền xử lý      : {len(df_clean):,} dòng")
    print(f"  Grid                 : {config.GRID_SIZE}×{config.GRID_SIZE} "
          f"({config.NUM_LAYERS} tầng)")
    print(f"  Ngưỡng query         : n ≥ {config.DENSITY_THRESHOLD}, "
          f"mean(tip) ≥ ${config.MEAN_THRESHOLD}")
    print(f"  Ô lá relevant        : {len(relevant_cells):,}")
    print(f"  Số cụm (clusters)    : {len(clusters)}")
    if clusters:
        print(f"  Cụm lớn nhất         : Cluster 0 ({clusters[0].size} ô)")
    print(f"\n  📊 Matplotlib output : {config.MATPLOTLIB_OUTPUT}")
    print(f"  🗺  Folium output    : {config.FOLIUM_OUTPUT}")
    print(f"\n{'='*60}")
    print("✓ Pipeline STING hoàn tất thành công!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
