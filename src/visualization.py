"""
visualization.py - Trực quan hoá kết quả thuật toán STING.

Cung cấp 2 chế độ hiển thị:
    1. Matplotlib: vẽ lưới 2D, tô màu các cụm (cluster) khác nhau.
    2. Folium: hiển thị trên bản đồ tương tác thực tế NYC, lưu ra HTML.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

import config
from src.clustering import Cluster
from src.grid import HierarchicalGrid
from src.utils import ensure_dir


class Visualizer:
    """
    Lớp trực quan hoá kết quả phân cụm STING.

    Attributes
    ----------
    grid : HierarchicalGrid
        Lưới phân cấp đã được nạp dữ liệu.
    clusters : list[Cluster]
        Danh sách các cụm tìm được từ RegionFormation.
    grid_size : int
        Kích thước lưới ở tầng đáy.
    lat_min, lat_max, lon_min, lon_max : float
        Giới hạn không gian bounding box.
    """

    def __init__(
        self,
        grid: HierarchicalGrid,
        clusters: list[Cluster],
    ):
        """
        Khởi tạo Visualizer.

        Parameters
        ----------
        grid : HierarchicalGrid
            Lưới phân cấp (dùng để truy cập thông tin thống kê ô).
        clusters : list[Cluster]
            Danh sách cụm kết quả từ BFS clustering.
        """
        self.grid = grid
        self.clusters = clusters
        self.grid_size = grid.grid_size
        self.lat_min = grid.lat_min
        self.lat_max = grid.lat_max
        self.lon_min = grid.lon_min
        self.lon_max = grid.lon_max

    # ==================================================================
    # 1. MATPLOTLIB — VẼ LƯỚI 2D VÀ TÔ MÀU CỤM
    # ==================================================================
    def plot_clusters_matplotlib(
        self,
        output_path: str = config.MATPLOTLIB_OUTPUT,
        figsize: tuple[int, int] = (14, 10),
        show: bool = True,
    ) -> None:
        """
        Vẽ lưới 2D bằng matplotlib, mỗi cụm một màu riêng biệt.

        - Các ô không thuộc cụm nào → màu nền xám nhạt.
        - Mỗi cụm → một màu riêng trên colormap.
        - Ô có dữ liệu nhưng không relevant → xám đậm hơn.

        Parameters
        ----------
        output_path : str
            Đường dẫn file ảnh đầu ra (.png).
        figsize : tuple[int, int]
            Kích thước figure.
        show : bool
            Có hiển thị figure trên màn hình không.
        """
        ensure_dir(output_path)

        # ----- Xây dựng ma trận màu -----
        # Giá trị mặc định = -1 (không thuộc cụm nào)
        grid_matrix = np.full((self.grid_size, self.grid_size), -1, dtype=int)

        # Gán ID cụm cho từng ô
        for cluster in self.clusters:
            for (r, c) in cluster.cells:
                grid_matrix[r, c] = cluster.cluster_id

        # ----- Tạo ma trận mật độ (density) làm nền -----
        density_matrix = np.zeros((self.grid_size, self.grid_size), dtype=float)
        leaf_layer = self.grid.get_layer(0)
        for (r, c), cell in leaf_layer.items():
            if cell.n > 0:
                density_matrix[r, c] = cell.n

        # ----- Vẽ -----
        fig, ax = plt.subplots(figsize=figsize)

        # Nền: mật độ dữ liệu (xám)
        ax.imshow(
            density_matrix,
            cmap="Greys",
            alpha=0.3,
            origin="lower",
            aspect="auto",
        )

        # Lớp phủ: cụm (cluster) với colormap riêng biệt
        num_clusters = len(self.clusters)
        if num_clusters > 0:
            # Tạo colormap rời rạc cho các cụm
            base_cmap = plt.get_cmap("tab20" if num_clusters <= 20 else "hsv")
            cluster_colors = [base_cmap(i / max(num_clusters, 1)) for i in range(num_clusters)]

            # Tạo ma trận RGBA
            overlay = np.zeros((self.grid_size, self.grid_size, 4))
            for cluster in self.clusters:
                color = cluster_colors[cluster.cluster_id]
                for (r, c) in cluster.cells:
                    overlay[r, c] = color

            ax.imshow(
                overlay,
                origin="lower",
                aspect="auto",
                alpha=0.8,
            )

            # Legend cho top cụm
            top_n = min(10, num_clusters)
            legend_patches = []
            for i in range(top_n):
                patch = mpatches.Patch(
                    color=cluster_colors[i],
                    label=f"Cluster {i} ({self.clusters[i].size} ô)",
                )
                legend_patches.append(patch)

            if num_clusters > top_n:
                legend_patches.append(
                    mpatches.Patch(color="grey", label=f"... +{num_clusters - top_n} cụm khác")
                )

            ax.legend(
                handles=legend_patches,
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
            )

        # ----- Nhãn và tiêu đề -----
        ax.set_title(
            f"STING Clustering — Hotspot Tip Amount (NYC Taxi)\n"
            f"Grid {self.grid_size}×{self.grid_size} | "
            f"{num_clusters} cụm | "
            f"Ngưỡng: n≥{config.DENSITY_THRESHOLD}, mean≥{config.MEAN_THRESHOLD}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Cột (Longitude →)", fontsize=11)
        ax.set_ylabel("Hàng (Latitude →)", fontsize=11)

        # Thêm đường lưới mờ
        ax.set_xticks(np.arange(0, self.grid_size, self.grid_size // 8))
        ax.set_yticks(np.arange(0, self.grid_size, self.grid_size // 8))
        ax.grid(True, alpha=0.2, linestyle="--")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] ✓ Đã lưu biểu đồ Matplotlib: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ==================================================================
    # 2. FOLIUM — BẢN ĐỒ TƯƠNG TÁC NYC
    # ==================================================================
    def plot_clusters_folium(
        self,
        output_path: str = config.FOLIUM_OUTPUT,
    ) -> None:
        """
        Hiển thị các cụm hotspot lên bản đồ tương tác thực tế NYC
        bằng thư viện Folium. Mỗi cụm một màu, mỗi ô là 1 hình chữ nhật.

        Parameters
        ----------
        output_path : str
            Đường dẫn file HTML đầu ra.
        """
        import folium

        ensure_dir(output_path)

        # Tâm bản đồ NYC
        center_lat = (self.lat_min + self.lat_max) / 2
        center_lon = (self.lon_min + self.lon_max) / 2

        # Kích thước mỗi ô (theo đơn vị toạ độ)
        cell_height = (self.lat_max - self.lat_min) / self.grid_size
        cell_width = (self.lon_max - self.lon_min) / self.grid_size

        # Tạo bản đồ nền
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="CartoDB positron",
        )

        # Bảng màu cho các cụm
        color_palette = [
            "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
            "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
            "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
        ]

        num_clusters = len(self.clusters)
        leaf_layer = self.grid.get_layer(0)

        for cluster in self.clusters:
            # Chọn màu (lặp lại nếu > 20 cụm)
            color = color_palette[cluster.cluster_id % len(color_palette)]

            for (r, c) in cluster.cells:
                # Chuyển chỉ số ô (row, col) → toạ độ thực (lat, lon)
                lat_sw = self.lat_min + r * cell_height       # góc Tây-Nam (lat)
                lat_ne = lat_sw + cell_height                  # góc Đông-Bắc (lat)
                lon_sw = self.lon_min + c * cell_width         # góc Tây-Nam (lon)
                lon_ne = lon_sw + cell_width                   # góc Đông-Bắc (lon)

                # Lấy thống kê ô để hiển thị popup
                cell = leaf_layer.get((r, c))
                popup_text = (
                    f"<b>Cluster {cluster.cluster_id}</b><br>"
                    f"Cell ({r}, {c})<br>"
                    f"n = {cell.n:,}<br>"
                    f"mean tip = ${cell.m:.2f}<br>"
                    f"std = ${cell.s:.2f}<br>"
                    f"min = ${cell.min_val:.2f}<br>"
                    f"max = ${cell.max_val:.2f}"
                ) if cell else f"Cluster {cluster.cluster_id}"

                # Vẽ hình chữ nhật cho ô
                folium.Rectangle(
                    bounds=[[lat_sw, lon_sw], [lat_ne, lon_ne]],
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.5,
                    weight=1,
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=f"Cluster {cluster.cluster_id} | ({r},{c})",
                ).add_to(m)

        # Thêm tiêu đề trên bản đồ
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 1000; background: rgba(255,255,255,0.9);
                    padding: 10px 20px; border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                    font-family: Arial, sans-serif;">
            <b>STING Hotspot Map — NYC Taxi Tip Amount</b><br>
            <span style="font-size:12px;">
                {num_clusters} cụm |
                Grid {self.grid_size}×{self.grid_size} |
                n ≥ {config.DENSITY_THRESHOLD}, mean ≥ ${config.MEAN_THRESHOLD}
            </span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Lưu file
        m.save(output_path)
        print(f"[Visualization] ✓ Đã lưu bản đồ Folium: {output_path}")
        print(f"  Mở file '{output_path}' trong trình duyệt để xem bản đồ tương tác.")
