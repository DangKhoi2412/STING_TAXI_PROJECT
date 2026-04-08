"""
statistics.py - Lớp StingCell và các công thức toán học của thuật toán STING.

Mỗi ô lưới (cell) trong STING lưu trữ 5 tham số thống kê trên biến mục tiêu
(tip_amount), KHÔNG lưu dữ liệu thô:
    • n   : số lượng điểm dữ liệu (count)
    • m   : trung bình (mean)
    • s   : độ lệch chuẩn (standard deviation)
    • min : giá trị nhỏ nhất
    • max : giá trị lớn nhất

Hai chế độ tính toán:
    1. Leaf (lá): tính trực tiếp từ mảng giá trị tip_amount.
    2. Parent (cha): gộp (aggregate) từ 4 ô con theo công thức chính xác
       trong bài báo gốc STING-1997 (Wang, Yang, Muntz).

Công thức Bottom-Up cho ô cha:
──────────────────────────────
    n = Σ nᵢ

    m = Σ(mᵢ × nᵢ) / n

              ┌─────────────────────────────────┐
    s = sqrt  │  Σ((sᵢ² + mᵢ²) × nᵢ) / n − m² │
              └─────────────────────────────────┘

    min = min(minᵢ)
    max = max(maxᵢ)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class StingCell:
    """
    Ô thống kê của thuật toán STING.

    Attributes
    ----------
    row : int
        Chỉ số hàng của ô trong lưới tại tầng hiện tại.
    col : int
        Chỉ số cột của ô trong lưới tại tầng hiện tại.
    layer : int
        Tầng (layer) mà ô thuộc về (0 = đáy, tăng dần lên root).
    n : int
        Số lượng điểm dữ liệu rơi vào ô.
    m : float
        Trung bình (mean) của tip_amount trong ô.
    s : float
        Độ lệch chuẩn (std) của tip_amount trong ô.
    min_val : float
        Giá trị tip_amount nhỏ nhất trong ô.
    max_val : float
        Giá trị tip_amount lớn nhất trong ô.
    children : list[StingCell]
        Danh sách 4 ô con (rỗng nếu là ô lá).
    is_relevant : Optional[bool]
        Nhãn phân loại sau bước query: True/False/None (chưa đánh giá).
    """

    def __init__(self, row: int, col: int, layer: int):
        """
        Khởi tạo một ô STING rỗng (chưa có dữ liệu thống kê).

        Parameters
        ----------
        row : int
            Chỉ số hàng.
        col : int
            Chỉ số cột.
        layer : int
            Tầng phân cấp.
        """
        self.row = row
        self.col = col
        self.layer = layer

        # 5 tham số thống kê — mặc định = 0
        self.n: int = 0
        self.m: float = 0.0
        self.s: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")

        # Liên kết phân cấp
        self.children: list[StingCell] = []

        # Nhãn truy vấn (được gán ở bước Top-Down Query)
        self.is_relevant: Optional[bool] = None

    # ==================================================================
    # TÍNH TOÁN CHO Ô LÁ (LEAF) — từ dữ liệu thô
    # ==================================================================
    def compute_from_data(self, values: np.ndarray) -> None:
        """
        Tính 5 tham số thống kê trực tiếp từ mảng giá trị tip_amount.

        Dùng cho các ô ở tầng đáy (leaf cells) — nơi dữ liệu thô
        được băm (hash) vào.

        Parameters
        ----------
        values : np.ndarray
            Mảng 1-D chứa các giá trị tip_amount của những
            điểm dữ liệu rơi vào ô này.

        Notes
        -----
        Nếu mảng rỗng (không có điểm nào), toàn bộ tham số giữ nguyên
        giá trị mặc định (n=0).
        """
        if len(values) == 0:
            return

        self.n = len(values)
        self.m = float(np.mean(values))
        # Dùng ddof=0 (population std) khớp với công thức gốc của STING
        self.s = float(np.std(values, ddof=0))
        self.min_val = float(np.min(values))
        self.max_val = float(np.max(values))

    # ==================================================================
    # TÍNH TOÁN CHO Ô CHA (PARENT) — gộp từ 4 ô con (Bottom-Up)
    # ==================================================================
    def aggregate_from_children(self) -> None:
        """
        Gộp (aggregate) thống kê từ 4 ô con theo đúng công thức
        bài báo STING-1997.

        Công thức toán học
        ------------------
        Cho 4 ô con có tham số (nᵢ, mᵢ, sᵢ, minᵢ, maxᵢ):

            n = Σ nᵢ                                          (1)

            m = Σ(mᵢ × nᵢ) / n                                (2)

            s = sqrt( Σ((sᵢ² + mᵢ²) × nᵢ) / n  −  m² )       (3)

            min = min(minᵢ)                                    (4)

            max = max(maxᵢ)                                    (5)

        Giải thích công thức (3):
        ─────────────────────────
        Đây là công thức gộp phương sai từ nhiều nhóm (pooled variance).

        Với mỗi nhóm i, ta có:
            E[Xᵢ²] = sᵢ² + mᵢ²   (vì Var(X) = E[X²] − E[X]²)

        Nên:
            E[X²] tổng thể = Σ((sᵢ² + mᵢ²) × nᵢ) / n

        Và:
            s² = E[X²] − m²

        ⇒ s  = sqrt(E[X²] − m²)

        LƯU Ý: Tuyệt đối KHÔNG tính trung bình cộng đơn giản của sᵢ,
        vì điều đó sai về mặt toán học khi các nhóm có giá trị trung bình
        (mᵢ) khác nhau.

        Raises
        ------
        ValueError
            Nếu danh sách children rỗng.
        """
        if not self.children:
            raise ValueError(
                f"[StingCell] Ô ({self.row}, {self.col}) tầng {self.layer} "
                "không có ô con để aggregate."
            )

        # Lọc bỏ các ô con rỗng (n == 0) — chúng không đóng góp gì
        active_children = [c for c in self.children if c.n > 0]

        if not active_children:
            # Tất cả ô con đều rỗng → ô cha cũng rỗng
            self.n = 0
            self.m = 0.0
            self.s = 0.0
            self.min_val = float("inf")
            self.max_val = float("-inf")
            return

        # --- (1) Tổng số điểm: n = Σ nᵢ ---
        self.n = sum(c.n for c in active_children)

        # --- (2) Trung bình gộp: m = Σ(mᵢ × nᵢ) / n ---
        weighted_sum = sum(c.m * c.n for c in active_children)
        self.m = weighted_sum / self.n

        # --- (3) Độ lệch chuẩn gộp (pooled std) ---
        #     E[X²] = Σ((sᵢ² + mᵢ²) × nᵢ) / n
        #     s = sqrt(E[X²] − m²)
        sum_sq = sum((c.s ** 2 + c.m ** 2) * c.n for c in active_children)
        variance = sum_sq / self.n - self.m ** 2

        # Bảo vệ chống lỗi số học (floating-point) khi variance âm rất nhỏ
        if variance < 0:
            variance = 0.0

        self.s = math.sqrt(variance)

        # --- (4) Min gộp ---
        self.min_val = min(c.min_val for c in active_children)

        # --- (5) Max gộp ---
        self.max_val = max(c.max_val for c in active_children)

    # ==================================================================
    # TIỆN ÍCH
    # ==================================================================
    def is_empty(self) -> bool:
        """Kiểm tra ô có rỗng (không chứa điểm dữ liệu nào) không."""
        return self.n == 0

    def __repr__(self) -> str:
        """Biểu diễn chuỗi ngắn gọn cho mục đích debug."""
        return (
            f"StingCell(layer={self.layer}, row={self.row}, col={self.col}, "
            f"n={self.n}, m={self.m:.3f}, s={self.s:.3f}, "
            f"min={self.min_val:.3f}, max={self.max_val:.3f}, "
            f"relevant={self.is_relevant})"
        )
