"""
utils.py - Các hàm tiện ích dùng chung cho toàn bộ dự án STING.

Bao gồm:
    • Timer: đo thời gian chạy từng bước trong pipeline.
    • Logger: in log có định dạng thống nhất.
    • ensure_dir: tạo thư mục output nếu chưa tồn tại.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator


# ==================================================================
# ĐO THỜI GIAN CHẠY (CONTEXT MANAGER)
# ==================================================================
@contextmanager
def timer(step_name: str) -> Generator[None, None, None]:
    """
    Context manager để đo thời gian thực thi một bước.

    Sử dụng:
    --------
        with timer("Tiền xử lý"):
            preprocessor.transform(df)

    Sẽ in ra:
        ⏱ [Tiền xử lý] Hoàn tất trong 1.23 giây

    Parameters
    ----------
    step_name : str
        Tên bước (dùng để hiển thị trong log).
    """
    print(f"\n{'='*60}")
    print(f"▶ Bắt đầu: {step_name}")
    print(f"{'='*60}")

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    print(f"⏱ [{step_name}] Hoàn tất trong {elapsed:.2f} giây")


# ==================================================================
# TẠO THƯ MỤC OUTPUT
# ==================================================================
def ensure_dir(path: str) -> None:
    """
    Tạo thư mục (và các thư mục cha) nếu chưa tồn tại.

    Parameters
    ----------
    path : str
        Đường dẫn thư mục cần tạo.
    """
    directory = os.path.dirname(path) if "." in os.path.basename(path) else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[Utils] Đã tạo thư mục: {directory}")


# ==================================================================
# IN PHÂN CÁCH
# ==================================================================
def print_header(title: str) -> None:
    """In tiêu đề phân cách giữa các bước pipeline."""
    print(f"\n{'#'*60}")
    print(f"# {title}")
    print(f"{'#'*60}\n")
