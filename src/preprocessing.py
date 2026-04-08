"""
preprocessing.py - Module tiền xử lý dữ liệu.

Thực hiện Step 2 trong pipeline:
  • Lọc toạ độ nằm trong bounding box NYC (Lat: 40.5–40.9, Lon: -74.25–-73.7).
  • Loại bỏ các chuyến đi có trip_distance <= 0 hoặc fare_amount <= 0.
  • Lọc chỉ giữ thanh toán bằng thẻ tín dụng (payment_type == 1).
  • Trích xuất đặc trưng giờ (hour) từ cột tpep_pickup_datetime.
"""

import pandas as pd
import config


class Preprocessor:
    """
    Lớp tiền xử lý dữ liệu NYC Taxi.

    Pipeline xử lý theo thứ tự:
        1. Loại bỏ giá trị thiếu (NaN).
        2. Lọc toạ độ theo bounding box NYC.
        3. Loại bỏ chuyến đi có khoảng cách hoặc giá vé không hợp lệ.
        4. Lọc chỉ giữ thanh toán bằng thẻ tín dụng.
        5. Trích xuất đặc trưng thời gian (hour) từ datetime.

    Attributes
    ----------
    lat_min, lat_max : float
        Giới hạn vĩ độ (latitude) hợp lệ.
    lon_min, lon_max : float
        Giới hạn kinh độ (longitude) hợp lệ.
    """

    def __init__(
        self,
        lat_min: float = config.LAT_MIN,
        lat_max: float = config.LAT_MAX,
        lon_min: float = config.LON_MIN,
        lon_max: float = config.LON_MAX,
    ):
        """
        Khởi tạo Preprocessor với các giới hạn bounding box.

        Parameters
        ----------
        lat_min : float
            Vĩ độ nhỏ nhất (mặc định 40.5).
        lat_max : float
            Vĩ độ lớn nhất (mặc định 40.9).
        lon_min : float
            Kinh độ nhỏ nhất (mặc định -74.25).
        lon_max : float
            Kinh độ lớn nhất (mặc định -73.7).
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    # ------------------------------------------------------------------
    # Pipeline chính
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline tiền xử lý theo đúng thứ tự.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame thô từ DataLoader.

        Returns
        -------
        pd.DataFrame
            DataFrame đã được làm sạch và bổ sung cột 'hour'.
        """
        initial_count = len(df)
        print(f"[Preprocessor] Bắt đầu tiền xử lý — {initial_count:,} dòng")

        # Bước 1: Xoá các dòng có giá trị thiếu
        df = self._drop_missing(df)

        # Bước 2: Lọc toạ độ theo bounding box NYC
        df = self._filter_bounding_box(df)

        # Bước 3: Loại bỏ chuyến đi không hợp lệ (distance <= 0, fare <= 0)
        df = self._filter_invalid_trips(df)

        # Bước 4: Lọc chỉ giữ thanh toán bằng thẻ tín dụng
        df = self._filter_credit_card(df)

        # Bước 5: Trích xuất đặc trưng giờ từ cột datetime
        df = self._extract_hour(df)

        final_count = len(df)
        removed = initial_count - final_count
        print(f"[Preprocessor] Hoàn tất — còn {final_count:,} dòng "
              f"(đã loại {removed:,}, tỷ lệ giữ lại: {final_count / initial_count * 100:.1f}%)\n")

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Bước 1: Xoá dòng thiếu giá trị
    # ------------------------------------------------------------------
    def _drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ tất cả các dòng chứa ít nhất 1 giá trị NaN.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame sau khi xoá các dòng NaN.
        """
        before = len(df)
        df = df.dropna()
        after = len(df)
        print(f"  [1/5] Xoá dòng NaN: {before - after:,} dòng bị loại")
        return df

    # ------------------------------------------------------------------
    # Bước 2: Lọc bounding box NYC
    # ------------------------------------------------------------------
    def _filter_bounding_box(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chỉ giữ lại các điểm nằm trong bounding box NYC.

        Điều kiện:
            lat_min <= pickup_latitude  <= lat_max
            lon_min <= pickup_longitude <= lon_max

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame chỉ chứa toạ độ hợp lệ trong NYC.
        """
        before = len(df)
        mask = (
            (df["pickup_latitude"] >= self.lat_min)
            & (df["pickup_latitude"] <= self.lat_max)
            & (df["pickup_longitude"] >= self.lon_min)
            & (df["pickup_longitude"] <= self.lon_max)
        )
        df = df[mask]
        after = len(df)
        print(f"  [2/5] Lọc bounding box NYC "
              f"(Lat: {self.lat_min}–{self.lat_max}, "
              f"Lon: {self.lon_min}–{self.lon_max}): "
              f"{before - after:,} dòng bị loại")
        return df

    # ------------------------------------------------------------------
    # Bước 3: Loại bỏ chuyến đi không hợp lệ
    # ------------------------------------------------------------------
    def _filter_invalid_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ các chuyến đi có giá trị không hợp lệ:
            - trip_distance <= 0
            - fare_amount  <= 0

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame chỉ chứa các chuyến đi hợp lệ.
        """
        before = len(df)
        mask = (df["trip_distance"] > 0) & (df["fare_amount"] > 0)
        df = df[mask]
        after = len(df)
        print(f"  [3/5] Loại chuyến đi không hợp lệ "
              f"(distance <= 0 hoặc fare <= 0): "
              f"{before - after:,} dòng bị loại")
        return df

    # ------------------------------------------------------------------
    # Bước 4: Lọc thanh toán bằng thẻ tín dụng
    # ------------------------------------------------------------------
    def _filter_credit_card(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chỉ giữ lại các chuyến đi thanh toán bằng thẻ tín dụng.

        Lý do:
            Dữ liệu TLC chỉ ghi nhận tip chính xác cho thanh toán
            bằng thẻ tín dụng (payment_type == 1). Thanh toán tiền mặt
            ghi tip_amount = 0.0. Nếu giữ lại những dòng này, giá trị
            mean(tip) của các ô sẽ bị kéo xuống thấp giả tạo, dẫn đến
            kết quả STING Query thiếu chính xác.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame chỉ chứa thanh toán bằng thẻ tín dụng.
        """
        before = len(df)
        df = df[df["payment_type"] == config.PAYMENT_TYPE_CREDIT]
        after = len(df)
        print(f"  [4/5] Lọc thanh toán thẻ tín dụng "
              f"(payment_type == {config.PAYMENT_TYPE_CREDIT}): "
              f"{before - after:,} dòng bị loại")
        return df

    # ------------------------------------------------------------------
    # Bước 5: Trích xuất đặc trưng giờ
    # ------------------------------------------------------------------
    def _extract_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất giờ (0–23) từ cột 'tpep_pickup_datetime' và
        lưu vào cột mới tên 'hour'.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame đã bổ sung cột 'hour'.
        """
        df = df.copy()  # Tránh SettingWithCopyWarning
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["tpep_pickup_datetime"], errors="coerce"
        )
        df["hour"] = df["tpep_pickup_datetime"].dt.hour
        print(f"  [5/5] Trích xuất đặc trưng 'hour' từ datetime — OK")
        return df
