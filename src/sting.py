"""
sting.py - Logic truy vấn Top-Down của thuật toán STING.

Thực hiện Step 5 trong pipeline:
    • Bắt đầu từ tầng cao nhất (root-level).
    • Kiểm tra từng ô theo điều kiện truy vấn (density + mean threshold).
    • CẮT TỈA (Pruning): nếu ô cha KHÔNG thoả mãn → bỏ qua toàn bộ nhánh
      con bên dưới, KHÔNG duyệt tiếp.
    • Chỉ đi sâu xuống 4 ô con khi ô cha thoả mãn điều kiện.
    • Kết quả: danh sách toạ độ (row, col) các ô lá (Layer 0) được đánh
      là "relevant" (hotspot).

Tham khảo: STING: A Statistical Information Grid Approach to Spatial
Data Mining (Wang, Yang, Muntz, 1997) — Section 3: Query Processing.
"""

from __future__ import annotations

from typing import Callable

import config
from src.grid import HierarchicalGrid
from src.statistics import StingCell


class StingQuery:
    """
    Bộ truy vấn Top-Down cho thuật toán STING.

    Thuật toán duyệt cây Quad-tree từ trên xuống (top → bottom):
    ──────────────────────────────────────────────────────────────
        1. Bắt đầu ở tầng cao nhất.
        2. Với mỗi ô, kiểm tra điều kiện lọc (relevance condition).
        3. Nếu ô thoả mãn → đánh dấu 'relevant', đi xuống 4 ô con.
        4. Nếu ô KHÔNG thoả mãn → đánh dấu 'not relevant', CẮT TỈA
           toàn bộ nhánh con (Pruning) → tiết kiệm đáng kể thời gian.
        5. Khi đến tầng đáy (Layer 0), thu thập toạ độ các ô relevant.

    Attributes
    ----------
    grid : HierarchicalGrid
        Lưới phân cấp đã được nạp dữ liệu và tính thống kê.
    min_n : int
        Ngưỡng mật độ — ô phải chứa ít nhất min_n điểm dữ liệu.
    min_mean : float
        Ngưỡng trung bình — mean(tip_amount) phải ≥ min_mean.
    relevant_cells : list[tuple[int, int]]
        Danh sách toạ độ (row, col) các ô lá thoả mãn điều kiện.
    stats : dict
        Thống kê quá trình truy vấn (số ô đã duyệt, đã cắt tỉa...).
    """

    def __init__(
        self,
        grid: HierarchicalGrid,
        min_n: int = config.DENSITY_THRESHOLD,
        min_mean: float = config.MEAN_THRESHOLD,
    ):
        """
        Khởi tạo bộ truy vấn STING.

        Parameters
        ----------
        grid : HierarchicalGrid
            Lưới đã nạp dữ liệu (đã chạy feed_data + aggregate).
        min_n : int
            Ngưỡng mật độ tối thiểu (mặc định từ config).
        min_mean : float
            Ngưỡng trung bình tip_amount tối thiểu (mặc định từ config).
        """
        self.grid = grid
        self.min_n = min_n
        self.min_mean = min_mean

        # Kết quả truy vấn
        self.relevant_cells: list[tuple[int, int]] = []

        # Thống kê hiệu suất
        self.stats: dict = {
            "total_visited": 0,     # Tổng số ô đã duyệt
            "total_pruned": 0,      # Tổng số ô bị cắt tỉa (bỏ qua)
            "total_relevant": 0,    # Tổng số ô lá relevant
        }

    # ==================================================================
    # ĐIỀU KIỆN RELEVANCE
    # ==================================================================
    def _is_relevant(self, cell: StingCell) -> bool:
        """
        Kiểm tra một ô có thoả mãn điều kiện truy vấn hay không.

        Điều kiện (AND):
            1. cell.n    ≥ min_n      (đủ mật độ)
            2. cell.m    ≥ min_mean   (trung bình tip đủ lớn)

        Parameters
        ----------
        cell : StingCell
            Ô cần kiểm tra.

        Returns
        -------
        bool
            True nếu ô thoả mãn TẤT CẢ các điều kiện.

        Notes
        -----
        Người dùng có thể mở rộng điều kiện (ví dụ thêm ngưỡng std)
        bằng cách override phương thức này trong lớp con.
        """
        # Ô rỗng luôn không thoả mãn
        if cell.is_empty():
            return False

        return cell.n >= self.min_n and cell.m >= self.min_mean

    # ==================================================================
    # TRUY VẤN TOP-DOWN (ENTRY POINT)
    # ==================================================================
    def execute(self) -> list[tuple[int, int]]:
        """
        Chạy truy vấn Top-Down trên toàn bộ cây Quad-tree.

        Quy trình:
            1. Lấy tầng cao nhất.
            2. Duyệt từng ô → kiểm tra relevance.
            3. Nếu relevant → đệ quy xuống 4 ô con.
            4. Nếu not relevant → Pruning (bỏ qua nhánh).
            5. Thu thập ô lá relevant ở Layer 0.

        Returns
        -------
        list[tuple[int, int]]
            Danh sách toạ độ (row, col) của các ô lá relevant.
        """
        # Reset kết quả từ lần chạy trước (nếu có)
        self.relevant_cells = []
        self.stats = {"total_visited": 0, "total_pruned": 0, "total_relevant": 0}

        top_layer_index = self.grid.num_layers - 1
        top_layer = self.grid.get_layer(top_layer_index)

        print(f"[STING Query] Bắt đầu truy vấn Top-Down")
        print(f"  Điều kiện: n ≥ {self.min_n}, mean(tip) ≥ {self.min_mean}")
        print(f"  Tầng bắt đầu: {top_layer_index} "
              f"({self.grid.layer_size(top_layer_index)}×"
              f"{self.grid.layer_size(top_layer_index)} ô)")

        # Duyệt từng ô ở tầng cao nhất
        for cell in top_layer.values():
            self._traverse(cell, current_layer=top_layer_index)

        self.stats["total_relevant"] = len(self.relevant_cells)

        # Báo cáo kết quả
        print(f"\n[STING Query] ✓ Hoàn tất truy vấn")
        print(f"  Ô đã duyệt : {self.stats['total_visited']:,}")
        print(f"  Ô bị cắt tỉa: {self.stats['total_pruned']:,}")
        print(f"  Ô lá relevant: {self.stats['total_relevant']:,} / "
              f"{self.grid.grid_size ** 2:,}\n")

        return self.relevant_cells

    # ==================================================================
    # ĐỆ QUY DUYỆT CÂY (TOP → BOTTOM)
    # ==================================================================
    def _traverse(self, cell: StingCell, current_layer: int) -> None:
        """
        Duyệt đệ quy một ô và các nhánh con bên dưới.

        Logic Pruning (cắt tỉa):
        ─────────────────────────
        Tại mỗi ô, ta kiểm tra điều kiện relevance:

        • Nếu ô THOẢ MÃN (relevant):
            → Đánh dấu is_relevant = True.
            → Nếu là ô lá (Layer 0): thêm toạ độ vào kết quả.
            → Nếu chưa phải ô lá: đệ quy xuống 4 ô con.

        • Nếu ô KHÔNG thoả mãn (not relevant):
            → Đánh dấu is_relevant = False.
            → LẬP TỨC BỎ QUA toàn bộ nhánh con bên dưới.
            → Đây chính là bước PRUNING — giúp STING nhanh hơn rất
              nhiều so với quét toàn bộ lưới.

        Parameters
        ----------
        cell : StingCell
            Ô đang duyệt.
        current_layer : int
            Chỉ số tầng hiện tại của ô.
        """
        self.stats["total_visited"] += 1

        # ── Kiểm tra điều kiện relevance ──
        if not self._is_relevant(cell):
            # ╔══════════════════════════════════════════════════╗
            # ║  PRUNING: Ô không thoả mãn → cắt tỉa nhánh.   ║
            # ║  Không cần duyệt bất kỳ ô con nào bên dưới.    ║
            # ║  Tất cả các ô con (và cháu, chắt...) được coi  ║
            # ║  là not-relevant mà KHÔNG cần kiểm tra.         ║
            # ╚══════════════════════════════════════════════════╝
            cell.is_relevant = False
            self.stats["total_pruned"] += 1
            return

        # ── Ô thoả mãn điều kiện ──
        cell.is_relevant = True

        if current_layer == 0:
            # ── Đã đến tầng đáy (leaf) → thu thập kết quả ──
            self.relevant_cells.append((cell.row, cell.col))
        else:
            # ── Chưa đến đáy → đệ quy xuống 4 ô con ──
            for child in cell.children:
                self._traverse(child, current_layer - 1)

    # ==================================================================
    # TIỆN ÍCH
    # ==================================================================
    def get_relevant_cells(self) -> list[tuple[int, int]]:
        """Trả về danh sách toạ độ ô relevant (sau khi đã chạy execute)."""
        return self.relevant_cells

    def get_stats(self) -> dict:
        """Trả về thống kê quá trình truy vấn."""
        return self.stats
