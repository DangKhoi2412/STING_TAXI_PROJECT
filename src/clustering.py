"""
clustering.py - Gom cụm (Region Formation) bằng BFS trên các ô relevant.

Thực hiện Step 6 trong pipeline:
    • Nhận danh sách toạ độ (row, col) các ô lá tại Layer 0 đã được
      đánh dấu "relevant" bởi bước Top-Down Query.
    • Dùng thuật toán BFS (Breadth-First Search) để "loang" từ mỗi ô
      chưa được thăm, gom tất cả các ô liền kề vào cùng một cụm.
    • Kết quả: danh sách các Cluster, mỗi Cluster là tập hợp toạ độ.

Định nghĩa liền kề (Adjacency):
────────────────────────────────
    Sử dụng 8-way connectivity (Moore neighborhood).
    Mỗi ô (r, c) có tối đa 8 ô hàng xóm:

        (r-1, c-1)  (r-1, c)  (r-1, c+1)      NW   N   NE
        (r,   c-1)  [r,  c ]  (r,   c+1)       W   [X]   E
        (r+1, c-1)  (r+1, c)  (r+1, c+1)      SW   S   SE

Tham khảo: STING-1997, Section 4: Discovering Regions.
"""

from __future__ import annotations

from collections import deque


# 8 hướng di chuyển (Moore neighborhood):
#   (Δrow, Δcol) cho N, S, E, W, NE, NW, SE, SW
DIRECTIONS_8 = [
    (-1,  0),   # Bắc  (N)
    ( 1,  0),   # Nam  (S)
    ( 0,  1),   # Đông (E)
    ( 0, -1),   # Tây  (W)
    (-1,  1),   # Đông-Bắc (NE)
    (-1, -1),   # Tây-Bắc  (NW)
    ( 1,  1),   # Đông-Nam (SE)
    ( 1, -1),   # Tây-Nam  (SW)
]


class Cluster:
    """
    Một cụm (cluster) chứa tập hợp các ô liền kề relevant.

    Attributes
    ----------
    cluster_id : int
        Mã định danh duy nhất của cụm.
    cells : list[tuple[int, int]]
        Danh sách toạ độ (row, col) của các ô thuộc cụm.
    """

    def __init__(self, cluster_id: int):
        """
        Khởi tạo một cụm rỗng.

        Parameters
        ----------
        cluster_id : int
            Số thứ tự / mã định danh của cụm.
        """
        self.cluster_id = cluster_id
        self.cells: list[tuple[int, int]] = []

    def add_cell(self, row: int, col: int) -> None:
        """Thêm một ô vào cụm."""
        self.cells.append((row, col))

    @property
    def size(self) -> int:
        """Số lượng ô trong cụm."""
        return len(self.cells)

    def __repr__(self) -> str:
        return f"Cluster(id={self.cluster_id}, size={self.size})"


class RegionFormation:
    """
    Lớp gom cụm (Region Formation) bằng BFS với 8-way adjacency.

    Quy trình BFS
    ──────────────
        1. Tạo tập hợp (set) chứa toạ độ các ô relevant → tra cứu O(1).
        2. Tạo tập visited để theo dõi ô đã thăm.
        3. Duyệt từng ô relevant chưa thăm:
            a. Khởi tạo Cluster mới + hàng đợi BFS.
            b. Đưa ô hiện tại vào hàng đợi.
            c. Khi hàng đợi chưa rỗng:
                - Lấy ô đầu hàng đợi ra.
                - Kiểm tra 8 ô hàng xóm.
                - Nếu hàng xóm nằm trong tập relevant VÀ chưa thăm
                  → thêm vào cụm + đẩy vào hàng đợi.
            d. Kết thúc → toàn bộ vùng liên thông đã được gom.

    Attributes
    ----------
    relevant_set : set[tuple[int, int]]
        Tập hợp toạ độ các ô relevant (để tra cứu nhanh O(1)).
    clusters : list[Cluster]
        Danh sách các cụm đã tìm được.
    """

    def __init__(self, relevant_cells: list[tuple[int, int]]):
        """
        Khởi tạo RegionFormation.

        Parameters
        ----------
        relevant_cells : list[tuple[int, int]]
            Danh sách toạ độ (row, col) các ô relevant ở Layer 0,
            được trả về bởi StingQuery.execute().
        """
        # Chuyển sang set để tra cứu "ô này có relevant không?" trong O(1)
        self.relevant_set: set[tuple[int, int]] = set(relevant_cells)

        # Kết quả gom cụm
        self.clusters: list[Cluster] = []

    # ==================================================================
    # BFS GOM CỤM (ENTRY POINT)
    # ==================================================================
    def form_clusters(self) -> list[Cluster]:
        """
        Chạy BFS để gom tất cả các ô relevant thành các cụm liên thông.

        Returns
        -------
        list[Cluster]
            Danh sách các cụm. Mỗi cụm chứa toạ độ các ô liền kề
            thuộc cùng một vùng hotspot.
        """
        # Reset kết quả từ lần chạy trước
        self.clusters = []

        # Tập hợp các ô đã được thăm
        visited: set[tuple[int, int]] = set()

        # Bộ đếm ID cụm
        cluster_id = 0

        print(f"[Clustering] Bắt đầu BFS gom cụm — {len(self.relevant_set):,} ô relevant")
        print(f"[Clustering] Adjacency: 8-way (Moore neighborhood)")

        # ── Duyệt từng ô relevant ──
        for cell in self.relevant_set:
            if cell in visited:
                # Ô này đã thuộc về một cụm trước đó → bỏ qua
                continue

            # ── Phát hiện cụm mới → khởi chạy BFS ──
            new_cluster = self._bfs(cell, visited, cluster_id)
            self.clusters.append(new_cluster)
            cluster_id += 1

        # Sắp xếp cụm theo kích thước giảm dần (cụm lớn nhất trước)
        self.clusters.sort(key=lambda c: c.size, reverse=True)

        # Gán lại ID theo thứ tự kích thước
        for i, cluster in enumerate(self.clusters):
            cluster.cluster_id = i

        # Báo cáo kết quả
        self._print_report()

        return self.clusters

    # ==================================================================
    # BFS — LOANG TỪ MỘT Ô
    # ==================================================================
    def _bfs(
        self,
        start: tuple[int, int],
        visited: set[tuple[int, int]],
        cluster_id: int,
    ) -> Cluster:
        """
        Thực hiện BFS (Breadth-First Search) từ ô `start` để tìm
        toàn bộ vùng liên thông 8-way của các ô relevant.

        Quy trình chi tiết:
        ────────────────────
        1. Đưa ô start vào hàng đợi (queue) và đánh dấu visited.
        2. Lặp cho đến khi hàng đợi rỗng:
            a. Lấy ô (r, c) ra khỏi đầu hàng đợi.
            b. Thêm (r, c) vào Cluster hiện tại.
            c. Kiểm tra 8 ô hàng xóm (nr, nc):
                - Nếu (nr, nc) ∈ relevant_set VÀ (nr, nc) ∉ visited:
                    → Đánh dấu visited.
                    → Đẩy vào hàng đợi.
            d. Quay lại bước 2.
        3. Kết quả: Cluster chứa TẤT CẢ các ô liên thông với start.

        Parameters
        ----------
        start : tuple[int, int]
            Toạ độ (row, col) của ô bắt đầu loang.
        visited : set[tuple[int, int]]
            Tập hợp các ô đã được thăm (dùng chung giữa các lần BFS).
        cluster_id : int
            ID của cụm đang xây dựng.

        Returns
        -------
        Cluster
            Cụm chứa toàn bộ vùng liên thông.
        """
        cluster = Cluster(cluster_id)

        # Hàng đợi BFS — dùng deque cho hiệu năng O(1) khi popleft
        queue: deque[tuple[int, int]] = deque()

        # Khởi tạo: thêm ô xuất phát
        queue.append(start)
        visited.add(start)

        while queue:
            r, c = queue.popleft()

            # Thêm ô hiện tại vào cụm
            cluster.add_cell(r, c)

            # ── Kiểm tra 8 ô hàng xóm (Moore neighborhood) ──
            #
            #   NW  N  NE        (-1,-1) (-1,0) (-1,+1)
            #    W  X  E    →    ( 0,-1) [r, c] ( 0,+1)
            #   SW  S  SE        (+1,-1) (+1,0) (+1,+1)
            #
            for dr, dc in DIRECTIONS_8:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                # Điều kiện để thêm hàng xóm vào hàng đợi:
                #   1. Ô hàng xóm phải thuộc tập relevant (là hotspot)
                #   2. Ô hàng xóm chưa được thăm (tránh lặp vô hạn)
                if neighbor in self.relevant_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return cluster

    # ==================================================================
    # BÁO CÁO KẾT QUẢ
    # ==================================================================
    def _print_report(self) -> None:
        """In báo cáo tổng hợp kết quả gom cụm."""
        print(f"\n[Clustering] ✓ Hoàn tất — Tìm thấy {len(self.clusters)} cụm")

        if not self.clusters:
            print("  Không có cụm nào được tìm thấy.\n")
            return

        # In top 10 cụm lớn nhất
        top_n = min(10, len(self.clusters))
        print(f"  Top {top_n} cụm lớn nhất:")
        for cluster in self.clusters[:top_n]:
            print(f"    Cluster {cluster.cluster_id}: {cluster.size} ô")

        total_cells = sum(c.size for c in self.clusters)
        print(f"  Tổng ô trong tất cả cụm: {total_cells:,}")
        print(f"  Kích thước trung bình: {total_cells / len(self.clusters):.1f} ô/cụm\n")

    # ==================================================================
    # TIỆN ÍCH
    # ==================================================================
    def get_clusters(self) -> list[Cluster]:
        """Trả về danh sách các cụm (sau khi đã chạy form_clusters)."""
        return self.clusters

    def get_largest_cluster(self) -> Cluster | None:
        """Trả về cụm lớn nhất, hoặc None nếu chưa có cụm nào."""
        return self.clusters[0] if self.clusters else None
