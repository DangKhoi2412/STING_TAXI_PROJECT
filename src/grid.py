"""
grid.py - Xây dựng lưới phân cấp Quad-tree cho thuật toán STING.

Cấu trúc phân cấp:
───────────────────
    • Tầng 0 (bottom): grid_size × grid_size ô (ô lá - leaf cells).
    • Tầng 1:           (grid_size/2) × (grid_size/2).
    • Tầng 2:           (grid_size/4) × (grid_size/4).
    • ...
    • Tầng (num_layers - 1): tầng cao nhất (root-level).

    Mỗi ô cha ở tầng k+1 chứa chính xác 4 ô con ở tầng k,
    tương ứng 4 góc: trên-trái, trên-phải, dưới-trái, dưới-phải.

Kỹ thuật Spatial Hashing (băm không gian):
──────────────────────────────────────────
    Thay vì dùng vòng lặp for từng điểm dữ liệu, module này sử dụng
    phép tính vectorized của NumPy để ánh xạ (lat, lon) → (row, col)
    trên toàn bộ DataFrame cùng lúc, đảm bảo hiệu năng cao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.statistics import StingCell


class HierarchicalGrid:
    """
    Lưới phân cấp Quad-tree cho thuật toán STING.

    Attributes
    ----------
    grid_size : int
        Kích thước lưới ở tầng đáy (ví dụ 64 → 64×64 ô).
    num_layers : int
        Số tầng phân cấp (bao gồm cả tầng đáy).
    lat_min, lat_max : float
        Giới hạn vĩ độ của vùng không gian.
    lon_min, lon_max : float
        Giới hạn kinh độ của vùng không gian.
    layers : list[dict[tuple[int, int], StingCell]]
        Danh sách các tầng. Mỗi tầng là một dict ánh xạ
        (row, col) → StingCell.

    Notes
    -----
    Quy ước đánh số tầng:
        - layers[0] = tầng đáy (leaf, grid_size × grid_size)
        - layers[-1] = tầng cao nhất (root-level)
    """

    def __init__(
        self,
        grid_size: int = config.GRID_SIZE,
        num_layers: int = config.NUM_LAYERS,
        lat_min: float = config.LAT_MIN,
        lat_max: float = config.LAT_MAX,
        lon_min: float = config.LON_MIN,
        lon_max: float = config.LON_MAX,
    ):
        """
        Khởi tạo lưới phân cấp.

        Parameters
        ----------
        grid_size : int
            Số ô mỗi chiều ở tầng đáy. Phải là luỹ thừa của 2
            và đủ lớn để chia đều qua num_layers tầng.
        num_layers : int
            Số tầng phân cấp.
        lat_min, lat_max : float
            Phạm vi vĩ độ.
        lon_min, lon_max : float
            Phạm vi kinh độ.
        """
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        # Xác nhận grid_size đủ lớn để chia num_layers tầng
        self._validate_config()

        # Khởi tạo cấu trúc lưới rỗng cho tất cả các tầng
        self.layers: list[dict[tuple[int, int], StingCell]] = []
        self._build_empty_grid()

        # Thiết lập liên kết cha-con giữa các tầng
        self._link_parent_children()

    # ==================================================================
    # XÂY DỰNG LƯỚI RỖNG
    # ==================================================================
    def _validate_config(self) -> None:
        """
        Kiểm tra tính hợp lệ của cấu hình lưới.

        Tầng đáy có grid_size × grid_size ô. Mỗi tầng cao hơn chia đôi
        mỗi chiều, nên kích thước tầng k = grid_size / 2^k.
        Kích thước này phải ≥ 1 ở tầng cao nhất.

        Raises
        ------
        ValueError
            Nếu grid_size không đủ lớn cho số tầng yêu cầu.
        """
        top_size = self.grid_size // (2 ** (self.num_layers - 1))
        if top_size < 1:
            raise ValueError(
                f"[Grid] grid_size={self.grid_size} quá nhỏ cho "
                f"num_layers={self.num_layers}. "
                f"Kích thước tầng cao nhất = {top_size} (cần ≥ 1)."
            )
        print(f"[Grid] Cấu hình hợp lệ:")
        for k in range(self.num_layers):
            size_k = self.grid_size // (2 ** k)
            print(f"  Tầng {k}: {size_k} × {size_k} = {size_k * size_k} ô")

    def _build_empty_grid(self) -> None:
        """
        Tạo các StingCell rỗng cho mỗi tầng.

        Tầng k có kích thước: (grid_size / 2^k) × (grid_size / 2^k).
        """
        for k in range(self.num_layers):
            size_k = self.grid_size // (2 ** k)
            layer: dict[tuple[int, int], StingCell] = {}
            for r in range(size_k):
                for c in range(size_k):
                    layer[(r, c)] = StingCell(row=r, col=c, layer=k)
            self.layers.append(layer)

    def _link_parent_children(self) -> None:
        """
        Thiết lập liên kết cha ↔ con giữa các tầng kề nhau.

        Ô cha (r, c) ở tầng k+1 chứa 4 ô con ở tầng k tại vị trí:
            (2r,   2c  )  — trên-trái
            (2r,   2c+1)  — trên-phải
            (2r+1, 2c  )  — dưới-trái
            (2r+1, 2c+1)  — dưới-phải
        """
        for k in range(1, self.num_layers):
            parent_layer = self.layers[k]
            child_layer = self.layers[k - 1]

            for (pr, pc), parent_cell in parent_layer.items():
                # 4 ô con tương ứng
                children_keys = [
                    (2 * pr,     2 * pc),      # trên-trái
                    (2 * pr,     2 * pc + 1),  # trên-phải
                    (2 * pr + 1, 2 * pc),      # dưới-trái
                    (2 * pr + 1, 2 * pc + 1),  # dưới-phải
                ]
                for ck in children_keys:
                    if ck in child_layer:
                        parent_cell.children.append(child_layer[ck])

    # ==================================================================
    # NẠP DỮ LIỆU VÀO TẦNG ĐÁY (VECTORIZED SPATIAL HASHING)
    # ==================================================================
    def feed_data(self, df: pd.DataFrame) -> None:
        """
        Băm toạ độ (lat, lon) của toàn bộ DataFrame vào lưới tầng đáy,
        sau đó tính thống kê và gộp bottom-up lên các tầng trên.

        Kỹ thuật Vectorized Spatial Hashing
        ────────────────────────────────────
        Thay vì for-loop từng dòng, ta dùng NumPy vectorized:
            row_idx = floor((lat - lat_min) / (lat_max - lat_min) × grid_size)
            col_idx = floor((lon - lon_min) / (lon_max - lon_min) × grid_size)

        Sau đó clip vào phạm vi [0, grid_size - 1] để xử lý biên.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame đã qua tiền xử lý, chứa các cột:
            'pickup_latitude', 'pickup_longitude', và config.TARGET_VARIABLE.
        """
        target = config.TARGET_VARIABLE  # "tip_amount"

        print(f"\n[Grid] Đang nạp {len(df):,} điểm dữ liệu vào lưới đáy "
              f"({self.grid_size}×{self.grid_size})...")

        # ----------------------------------------------------------
        # BƯỚC 1: Vectorized spatial hashing — tính chỉ số (row, col)
        # ----------------------------------------------------------
        lat = df["pickup_latitude"].values   # mảng NumPy
        lon = df["pickup_longitude"].values

        # Chuẩn hoá toạ độ về khoảng [0, 1], nhân với grid_size, lấy phần nguyên
        row_indices = np.floor(
            (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.grid_size
        ).astype(int)
        col_indices = np.floor(
            (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.grid_size
        ).astype(int)

        # Clip để đảm bảo nằm trong [0, grid_size - 1]
        # (xử lý trường hợp điểm nằm ngay trên biên phải/trên)
        row_indices = np.clip(row_indices, 0, self.grid_size - 1)
        col_indices = np.clip(col_indices, 0, self.grid_size - 1)

        # ----------------------------------------------------------
        # BƯỚC 2: Gom nhóm (group) theo (row, col) bằng pandas
        # ----------------------------------------------------------
        # Tạo DataFrame tạm để groupby
        hashed = pd.DataFrame({
            "row": row_indices,
            "col": col_indices,
            "target": df[target].values,
        })

        grouped = hashed.groupby(["row", "col"])["target"]

        # ----------------------------------------------------------
        # BƯỚC 3: Tính thống kê cho từng ô lá (leaf cell)
        # ----------------------------------------------------------
        leaf_layer = self.layers[0]
        cell_count = 0

        for (r, c), group_values in grouped:
            cell = leaf_layer.get((r, c))
            if cell is not None:
                cell.compute_from_data(group_values.values)
                cell_count += 1

        non_empty = sum(1 for c in leaf_layer.values() if not c.is_empty())
        print(f"[Grid] Tầng đáy: {non_empty:,}/{len(leaf_layer):,} ô "
              f"có dữ liệu (non-empty)")

        # ----------------------------------------------------------
        # BƯỚC 4: Bottom-Up aggregation — gộp lên các tầng trên
        # ----------------------------------------------------------
        self._aggregate_bottom_up()

    # ==================================================================
    # BOTTOM-UP AGGREGATION
    # ==================================================================
    def _aggregate_bottom_up(self) -> None:
        """
        Gộp thống kê từ tầng đáy lên tầng cao nhất.

        Duyệt từ tầng 1 → tầng (num_layers - 1). Mỗi ô cha gọi
        aggregate_from_children() để tổng hợp 5 tham số thống kê
        từ 4 ô con theo công thức STING-1997.
        """
        for k in range(1, self.num_layers):
            parent_layer = self.layers[k]
            size_k = self.grid_size // (2 ** k)
            non_empty = 0

            for cell in parent_layer.values():
                cell.aggregate_from_children()
                if not cell.is_empty():
                    non_empty += 1

            print(f"[Grid] Tầng {k}: {non_empty:,}/{size_k * size_k:,} ô "
                  f"có dữ liệu (aggregated)")

        print("[Grid] ✓ Bottom-Up aggregation hoàn tất.\n")

    # ==================================================================
    # TIỆN ÍCH
    # ==================================================================
    def get_layer(self, layer_index: int) -> dict[tuple[int, int], StingCell]:
        """
        Lấy toàn bộ ô của một tầng cụ thể.

        Parameters
        ----------
        layer_index : int
            Chỉ số tầng (0 = đáy, num_layers-1 = cao nhất).

        Returns
        -------
        dict[tuple[int, int], StingCell]
            Dict ánh xạ (row, col) → StingCell.
        """
        if not (0 <= layer_index < self.num_layers):
            raise IndexError(
                f"[Grid] layer_index={layer_index} ngoài phạm vi "
                f"[0, {self.num_layers - 1}]."
            )
        return self.layers[layer_index]

    def get_cell(self, layer_index: int, row: int, col: int) -> StingCell | None:
        """
        Lấy một ô cụ thể theo tầng, hàng, cột.

        Returns
        -------
        StingCell | None
            Trả về ô tương ứng hoặc None nếu không tồn tại.
        """
        return self.layers[layer_index].get((row, col))

    def layer_size(self, layer_index: int) -> int:
        """Trả về kích thước mỗi chiều tại tầng layer_index."""
        return self.grid_size // (2 ** layer_index)

    def print_summary(self) -> None:
        """In thống kê tổng hợp ở tầng cao nhất (root-level)."""
        top = self.num_layers - 1
        top_layer = self.layers[top]
        print(f"[Grid] === TỔNG HỢP TẦNG CAO NHẤT (Tầng {top}) ===")
        for (r, c), cell in sorted(top_layer.items()):
            if not cell.is_empty():
                print(f"  ({r},{c}): n={cell.n:,}, "
                      f"mean={cell.m:.3f}, std={cell.s:.3f}, "
                      f"min={cell.min_val:.2f}, max={cell.max_val:.2f}")
