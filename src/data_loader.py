"""
data_loader.py - Module chịu trách nhiệm đọc và kiểm tra dữ liệu thô.

Thực hiện Step 1 trong pipeline:
  • Đọc file CSV bằng pandas (hỗ trợ giới hạn số dòng qua SAMPLE_ROWS).
  • Chỉ giữ lại các cột cần thiết (định nghĩa trong config.COLUMNS).
  • Kiểm tra và báo cáo giá trị thiếu (missing values).
"""

import pandas as pd
import config


class DataLoader:
    """
    Lớp đọc dữ liệu NYC Taxi từ file CSV.

    Attributes
    ----------
    file_path : str
        Đường dẫn tới file dữ liệu.
    sample_rows : int | None
        Số dòng tối đa cần đọc. None = đọc toàn bộ.
    columns : list[str]
        Danh sách cột cần giữ lại sau khi đọc.
    df : pd.DataFrame | None
        DataFrame chứa dữ liệu sau khi load (ban đầu là None).
    """

    def __init__(
        self,
        file_path: str = config.DATA_PATH,
        sample_rows: int | None = config.SAMPLE_ROWS,
        columns: list[str] | None = None,
    ):
        """
        Khởi tạo DataLoader.

        Parameters
        ----------
        file_path : str
            Đường dẫn tới file CSV.
        sample_rows : int | None
            Số dòng load tối đa (None = toàn bộ).
        columns : list[str] | None
            Danh sách cột cần giữ. Mặc định dùng config.COLUMNS.
        """
        self.file_path = file_path
        self.sample_rows = sample_rows
        self.columns = columns or config.COLUMNS
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Đọc dữ liệu
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """
        Đọc file CSV và chỉ giữ lại các cột được chỉ định.

        Returns
        -------
        pd.DataFrame
            DataFrame chứa dữ liệu đã được đọc.

        Raises
        ------
        FileNotFoundError
            Nếu file không tồn tại tại đường dẫn đã cung cấp.
        ValueError
            Nếu file thiếu một hoặc nhiều cột yêu cầu.
        """
        print(f"[DataLoader] Đang đọc dữ liệu từ: {self.file_path}")
        print(f"[DataLoader] Số dòng tối đa: {self.sample_rows or 'Toàn bộ'}")

        # Đọc CSV với giới hạn số dòng (nrows)
        self.df = pd.read_csv(self.file_path, nrows=self.sample_rows)

        print(f"[DataLoader] Đã đọc xong: {self.df.shape[0]:,} dòng × {self.df.shape[1]} cột")

        # Kiểm tra xem các cột cần thiết có tồn tại trong dataset không
        self._validate_columns()

        # Chỉ giữ lại các cột cần thiết
        self.df = self.df[self.columns]
        print(f"[DataLoader] Giữ lại {len(self.columns)} cột: {self.columns}")

        return self.df

    # ------------------------------------------------------------------
    # Kiểm tra giá trị thiếu
    # ------------------------------------------------------------------
    def check_missing_values(self) -> pd.Series:
        """
        Kiểm tra và báo cáo số lượng giá trị thiếu (NaN) trên từng cột.

        Returns
        -------
        pd.Series
            Series chứa số lượng missing value của từng cột.

        Raises
        ------
        RuntimeError
            Nếu chưa gọi load() trước khi kiểm tra.
        """
        if self.df is None:
            raise RuntimeError(
                "[DataLoader] Chưa có dữ liệu. Hãy gọi load() trước."
            )

        missing = self.df.isnull().sum()
        total = len(self.df)

        print("\n[DataLoader] === BÁO CÁO GIÁ TRỊ THIẾU ===")
        for col, count in missing.items():
            pct = count / total * 100
            status = "✗" if count > 0 else "✓"
            print(f"  {status} {col}: {count:,} ({pct:.2f}%)")
        print(f"  Tổng số dòng: {total:,}\n")

        return missing

    # ------------------------------------------------------------------
    # Tổng quan nhanh về dữ liệu
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """
        In tổng quan nhanh về DataFrame: dtypes, shape, describe().
        Hữu ích cho bước EDA (Exploratory Data Analysis).
        """
        if self.df is None:
            raise RuntimeError(
                "[DataLoader] Chưa có dữ liệu. Hãy gọi load() trước."
            )

        print("[DataLoader] === TỔNG QUAN DỮ LIỆU ===")
        print(f"  Shape : {self.df.shape}")
        print(f"  Dtypes:\n{self.df.dtypes.to_string()}\n")
        print(self.df.describe().to_string())
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _validate_columns(self) -> None:
        """
        Xác nhận rằng tất cả các cột yêu cầu đều tồn tại trong DataFrame.

        Raises
        ------
        ValueError
            Nếu có cột yêu cầu không có trong dataset.
        """
        missing_cols = set(self.columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(
                f"[DataLoader] Các cột sau KHÔNG tồn tại trong dataset: {missing_cols}. "
                f"Các cột hiện có: {list(self.df.columns)}"
            )
