import pygmt
import numpy as np
from numpy import sin, cos, tan, radians
import math
from scipy.spatial import distance

def grd_to_xyz(input_grd, output_xyz):
    with pygmt.clib.Session() as session:
        # 使用pygmt.grd2xyz進行轉換
        session.call_module("grd2xyz", f"{input_grd} > {output_xyz}")

def twd67_to_97(x, y):
    """_summary_


    Parameters
    ----------
    x : float
        x in TWD67 system
    y : float
        x in TWD67 system

    Returns
    -------
    x and y in TWD97 system
    """
    A = 0.00001549
    B = 0.000006521

    x_97 = x + 807.8 + A * x + B * y
    y_97 = y - 248.6 + A * y + B * x
    return x_97, y_97


def twd97_to_lonlat(x=174458.0, y=2525824.0):
    """
    Parameters
    ----------
    x : float
        TWD97 coord system. The default is 174458.0.
    y : float
        TWD97 coord system. The default is 2525824.0.
    Returns
    -------
    list
        [longitude, latitude]
    """

    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0

    e = math.pow((1 - math.pow(b, 2) / math.pow(a, 2)), 0.5)

    x -= dx
    y -= dy

    M = y / k0

    mu = M / (
        a
        * (1 - math.pow(e, 2) / 4 - 3 * math.pow(e, 4) / 64 - 5 * math.pow(e, 6) / 256)
    )
    e1 = (1.0 - pow((1 - pow(e, 2)), 0.5)) / (
        1.0 + math.pow((1.0 - math.pow(e, 2)), 0.5)
    )

    j1 = 3 * e1 / 2 - 27 * math.pow(e1, 3) / 32
    j2 = 21 * math.pow(e1, 2) / 16 - 55 * math.pow(e1, 4) / 32
    j3 = 151 * math.pow(e1, 3) / 96
    j4 = 1097 * math.pow(e1, 4) / 512

    fp = (
        mu
        + j1 * math.sin(2 * mu)
        + j2 * math.sin(4 * mu)
        + j3 * math.sin(6 * mu)
        + j4 * math.sin(8 * mu)
    )

    e2 = math.pow((e * a / b), 2)
    c1 = math.pow(e2 * math.cos(fp), 2)
    t1 = math.pow(math.tan(fp), 2)
    r1 = (
        a
        * (1 - math.pow(e, 2))
        / math.pow((1 - math.pow(e, 2) * math.pow(math.sin(fp), 2)), (3 / 2))
    )
    n1 = a / math.pow((1 - math.pow(e, 2) * math.pow(math.sin(fp), 2)), 0.5)
    d = x / (n1 * k0)

    q1 = n1 * math.tan(fp) / r1
    q2 = math.pow(d, 2) / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1, 2) - 9 * e2) * math.pow(d, 4) / 24
    q4 = (
        (
            61
            + 90 * t1
            + 298 * c1
            + 45 * math.pow(t1, 2)
            - 3 * math.pow(c1, 2)
            - 252 * e2
        )
        * math.pow(d, 6)
        / 720
    )
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * math.pow(d, 3) / 6
    q7 = (
        (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1, 2) + 8 * e2 + 24 * math.pow(t1, 2))
        * math.pow(d, 5)
        / 120
    )
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)

    lat = (lat * 180) / math.pi
    lon = (lon * 180) / math.pi
    return [lon, lat]


def find_nearest_point(target_point, points):
    """
    找到點集points中距離目標點target_point最近的點。

    參數：
    target_point: 目標點的坐標，一個包含兩個元素的列表或元組，例如 [x, y]。
    points: 點集，一個包含多個點坐標的二維數組，每個點為一個包含兩個元素的列表或元組，例如 [[x1, y1], [x2, y2], ...]。

    返回值：
    nearest_point: 距離目標點最近的點的坐標，一個包含兩個元素的列表，例如 [x_nearest, y_nearest]。
    """
    target_point = np.array(target_point)
    points = np.array(points)

    # 使用cdist計算距離矩陣
    distances = distance.cdist([target_point], points)

    # 找到距離最小的點的索引
    nearest_index = np.argmin(distances)

    # 返回距離最小的點的坐標
    nearest_point = points[nearest_index]

    return nearest_index, nearest_point


def get_unique_with_other_columns(group):
    # 獲取唯一值
    unique_value = group["station_name"].unique()[0]
    # 獲取其他columns的值
    other_columns_values = group.drop_duplicates(subset=["station_name"])
    return other_columns_values
