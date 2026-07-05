"""
迷宫生成与寻路 — 1:1 移植自 frame_53/DoAction.as

数据结构 (与原版一致):
  maze[x][y] = [ground, h_wall, v_wall]
    - [0] ground: 恒为 1
    - [1] h_wall: 1 = 该格子"下边"有墙
    - [2] v_wall: 1 = 该格子"左边"有墙
  外边界永远封闭 (drawMaze 单独绘制四条边界)。

AS2 越界语义: maze[越界][y][k] == 0 求值为 false (undefined != 0)。
本模块用受保护的 h_open/v_open 复刻该行为。

未到达格子的距离值: 原版为 undefined, 参与比较均为 false, 参与加法得 NaN。
本模块用 float('nan') 复刻 (Python 中 nan 的比较同样恒为 False)。
"""

import math

NAN = float("nan")
SQRT2 = 1.4142135623730951  # 原版字面量


# ---------------------------------------------------------------- 基础访问

def h_open(maze, x, y):
    """maze[x][y][1] == 0 (含 AS2 越界=false 语义)"""
    if 0 <= x < len(maze) and 0 <= y < len(maze[x]):
        return maze[x][y][1] == 0
    return False


def v_open(maze, x, y):
    """maze[x][y][2] == 0 (含 AS2 越界=false 语义)"""
    if 0 <= x < len(maze) and 0 <= y < len(maze[x]):
        return maze[x][y][2] == 0
    return False


def _d(distances, x, y):
    """读取距离图, 越界/未填充返回 NaN (比较恒 False)"""
    if 0 <= x < len(distances) and 0 <= y < len(distances[x]):
        v = distances[x][y]
        return NAN if v is None else v
    return NAN


# ---------------------------------------------------------------- createMaze

def create_maze(xsize, ysize, rng):
    """frame_53:1-42 — 随机模板法迷宫生成。

    tempmaze 为 (xsize+1)x(ysize+1) 的 random(4) 值:
      下墙: tempmaze[x][y+1]==2 或 tempmaze[x+1][y+1]==0
      左墙: tempmaze[x][y]==1   或 tempmaze[x][y+1]==3
    """
    tempmaze = [[rng.randrange(4) for _ in range(ysize + 1)]
                for _ in range(xsize + 1)]
    maze = []
    for x in range(xsize):
        col = []
        for y in range(ysize):
            has_h = tempmaze[x][y + 1] == 2 or tempmaze[x + 1][y + 1] == 0
            has_v = tempmaze[x][y] == 1 or tempmaze[x][y + 1] == 3
            col.append([1, 1 if has_h else 0, 1 if has_v else 0])
        maze.append(col)
    return maze


# ---------------------------------------------------------------- calcReachable

def calc_reachable(maze, startx, starty):
    """frame_53:43-96 — DFS(栈) 求连通区域。

    返回 (reachable 列表[{x,y}], reachable_index 二维表)。
    入栈顺序与原版一致: 左、右、上、下。
    """
    w = len(maze)
    h = len(maze[0])
    reachable_index = [[None] * h for _ in range(w)]
    visited = {}
    out = []
    stack = [(startx, starty)]
    while stack:
        cx, cy = stack.pop()
        reachable_index[cx][cy] = len(out)
        out.append({"x": cx, "y": cy, "used": False})
        visited[(cx, cy)] = True
        # 左
        if v_open(maze, cx, cy) and cx > 0:
            if (cx - 1, cy) not in visited:
                visited[(cx - 1, cy)] = True
                stack.append((cx - 1, cy))
        # 右
        if v_open(maze, cx + 1, cy) and cx < w - 1:
            if (cx + 1, cy) not in visited:
                visited[(cx + 1, cy)] = True
                stack.append((cx + 1, cy))
        # 上
        if h_open(maze, cx, cy - 1) and cy > 0:
            if (cx, cy - 1) not in visited:
                visited[(cx, cy - 1)] = True
                stack.append((cx, cy - 1))
        # 下
        if h_open(maze, cx, cy) and cy < h - 1:
            if (cx, cy + 1) not in visited:
                visited[(cx, cy + 1)] = True
                stack.append((cx, cy + 1))
    return out, reachable_index


# ---------------------------------------------------------------- findDeadEnds

def find_dead_ends(maze, reachable, max_penalty):
    """frame_53:97-170 — 死胡同惩罚图。

    返回二维表: None=不可达, 0=普通格, >0=死胡同惩罚值(1..MAXDEADENDPENALTY)。
    """
    w = len(maze)
    h = len(maze[0])
    de = [[None] * h for _ in range(w)]
    stack = []
    for cell in reachable:
        stack.append((cell["x"], cell["y"]))
        de[cell["x"]][cell["y"]] = 0

    def val(x, y):
        if 0 <= x < w and 0 <= y < h:
            return de[x][y]
        return None

    while stack:
        cx, cy = stack.pop()
        if not de[cx][cy]:  # None 或 0 (AS2: !undefined / !0 均为 true)
            nxt = None
            open_count = 0
            penalty = max_penalty
            # 左
            if v_open(maze, cx, cy) and cx > 0 and not val(cx - 1, cy):
                nxt = (cx - 1, cy)
                open_count += 1
            elif v_open(maze, cx, cy) and cx > 0:
                penalty = max(1, min(de[cx - 1][cy] - 1, penalty))
            # 右
            if v_open(maze, cx + 1, cy) and cx < w - 1 and not val(cx + 1, cy):
                nxt = (cx + 1, cy)
                open_count += 1
            elif v_open(maze, cx + 1, cy) and cx < w - 1:
                penalty = max(1, min(de[cx + 1][cy] - 1, penalty))
            # 上
            if h_open(maze, cx, cy - 1) and cy > 0 and not val(cx, cy - 1):
                nxt = (cx, cy - 1)
                open_count += 1
            elif h_open(maze, cx, cy - 1) and cy > 0:
                penalty = max(1, min(de[cx][cy - 1] - 1, penalty))
            # 下
            if h_open(maze, cx, cy) and cy < h - 1 and not val(cx, cy + 1):
                nxt = (cx, cy + 1)
                open_count += 1
            elif h_open(maze, cx, cy) and cy < h - 1:
                penalty = max(1, min(de[cx][cy + 1] - 1, penalty))

            if open_count == 1:
                de[cx][cy] = penalty
                stack.append(nxt)
            if open_count == 0:
                de[cx][cy] = penalty
    return de


# ---------------------------------------------------------------- calcDistances

def calc_distances(maze, startx, starty):
    """frame_53:171-264 — FIFO 泛洪距离图 (4 正向 + 4 斜向 sqrt2)。

    注意: 原版是先到先得的 BFS (斜向代价 sqrt2 但不重松弛),
    邻居检查顺序影响结果, 此处顺序与原版完全一致:
    左、右、上、下、左下、右下、左上、右上。
    """
    w = len(maze)
    h = len(maze[0])
    dist = [[NAN] * h for _ in range(w)]
    visited = {}
    queue = [(startx, starty)]
    head = 0
    dist[startx][starty] = 0.0
    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        visited[(cx, cy)] = True

        def try_add(nx, ny, cost):
            if (nx, ny) not in visited:
                visited[(nx, ny)] = True
                dist[nx][ny] = dist[cx][cy] + cost
                queue.append((nx, ny))

        # 左
        if v_open(maze, cx, cy) and cx > 0:
            try_add(cx - 1, cy, 1)
        # 右
        if v_open(maze, cx + 1, cy) and cx < w - 1:
            try_add(cx + 1, cy, 1)
        # 上
        if h_open(maze, cx, cy - 1) and cy > 0:
            try_add(cx, cy - 1, 1)
        # 下
        if h_open(maze, cx, cy) and cy < h - 1:
            try_add(cx, cy + 1, 1)
        # 左下
        if (h_open(maze, cx, cy) and v_open(maze, cx, cy)
                and h_open(maze, cx - 1, cy) and v_open(maze, cx, cy + 1)
                and cx > 0 and cy < h - 1):
            try_add(cx - 1, cy + 1, SQRT2)
        # 右下
        if (h_open(maze, cx, cy) and v_open(maze, cx + 1, cy)
                and h_open(maze, cx + 1, cy) and v_open(maze, cx + 1, cy + 1)
                and cx < w - 1 and cy < h - 1):
            try_add(cx + 1, cy + 1, SQRT2)
        # 左上
        if (v_open(maze, cx, cy) and h_open(maze, cx, cy - 1)
                and v_open(maze, cx, cy - 1) and h_open(maze, cx - 1, cy - 1)
                and cx > 0 and cy > 0):
            try_add(cx - 1, cy - 1, SQRT2)
        # 右上
        if (v_open(maze, cx + 1, cy) and h_open(maze, cx, cy - 1)
                and h_open(maze, cx + 1, cy - 1) and v_open(maze, cx + 1, cy - 1)
                and cx < w - 1 and cy > 0):
            try_add(cx + 1, cy - 1, SQRT2)
    return dist


# ---------------------------------------------------------------- 最短路径

def get_shortest_path_with_distances(maze, distances, startx, starty, endx, endy):
    """frame_53:270-338 — 从终点沿距离下降走回起点。

    返回 [{x,y}, ...]: path[0]=紧邻起点的第一步, path[-1]=终点 (不含起点)。
    检查顺序: 左下、右下、左上、右上、左、右、上、下 (与原版一致)。
    原版为 do-while, 起点即终点时也会推入一个元素。
    加入安全上限防止死循环 (原版依赖连通性保证)。
    """
    w = len(maze)
    h = len(maze[0])
    path = []
    cx, cy = endx, endy
    best = _d(distances, cx, cy)
    nx, ny = endx, endy
    safety = w * h * 4 + 8
    while True:
        path.append({"x": cx, "y": cy})
        # 左下
        if (h_open(maze, cx, cy) and v_open(maze, cx, cy)
                and h_open(maze, cx - 1, cy) and v_open(maze, cx, cy + 1)
                and cx > 0 and cy < h - 1 and _d(distances, cx - 1, cy + 1) < best):
            best = _d(distances, cx - 1, cy + 1)
            nx, ny = cx - 1, cy + 1
        # 右下
        if (h_open(maze, cx, cy) and v_open(maze, cx + 1, cy)
                and h_open(maze, cx + 1, cy) and v_open(maze, cx + 1, cy + 1)
                and cx < w - 1 and cy < h - 1 and _d(distances, cx + 1, cy + 1) < best):
            best = _d(distances, cx + 1, cy + 1)
            nx, ny = cx + 1, cy + 1
        # 左上
        if (v_open(maze, cx, cy) and h_open(maze, cx, cy - 1)
                and v_open(maze, cx, cy - 1) and h_open(maze, cx - 1, cy - 1)
                and cx > 0 and cy > 0 and _d(distances, cx - 1, cy - 1) < best):
            best = _d(distances, cx - 1, cy - 1)
            nx, ny = cx - 1, cy - 1
        # 右上
        if (v_open(maze, cx + 1, cy) and h_open(maze, cx, cy - 1)
                and h_open(maze, cx + 1, cy - 1) and v_open(maze, cx + 1, cy - 1)
                and cx < w - 1 and cy > 0 and _d(distances, cx + 1, cy - 1) < best):
            best = _d(distances, cx + 1, cy - 1)
            nx, ny = cx + 1, cy - 1
        # 左
        if v_open(maze, cx, cy) and cx > 0 and _d(distances, cx - 1, cy) < best:
            best = _d(distances, cx - 1, cy)
            nx, ny = cx - 1, cy
        # 右
        if v_open(maze, cx + 1, cy) and cx < w - 1 and _d(distances, cx + 1, cy) < best:
            best = _d(distances, cx + 1, cy)
            nx, ny = cx + 1, cy
        # 上
        if h_open(maze, cx, cy - 1) and cy > 0 and _d(distances, cx, cy - 1) < best:
            best = _d(distances, cx, cy - 1)
            nx, ny = cx, cy - 1
        # 下
        if h_open(maze, cx, cy) and cy < h - 1 and _d(distances, cx, cy + 1) < best:
            best = _d(distances, cx, cy + 1)
            nx, ny = cx, cy + 1

        if (nx == cx and ny == cy) or safety <= 0:
            # 无下降方向 (原版此处会死循环; 安全终止)
            break
        cx, cy = nx, ny
        safety -= 1
        if cx == startx and cy == starty:
            break
    path.reverse()
    return path


def get_shortest_path(maze, startx, starty, endx, endy):
    """frame_53:265-269"""
    dist = calc_distances(maze, startx, starty)
    return get_shortest_path_with_distances(maze, dist, startx, starty, endx, endy)


# ---------------------------------------------------------------- 梯度上升

def _gradient_walk(maze, value_fn, startx, starty, max_length):
    """followGradientPath* 的公共骨架 — 沿 value 上升方向走 (逃离)。

    检查顺序: 左下、右下、左上、右上、左、右、上、下 (与原版一致)。
    do-while: 即使原地不动也会推入一个元素。
    """
    w = len(maze)
    h = len(maze[0])
    path = []
    cx, cy = startx, starty
    best = value_fn(cx, cy)
    while True:
        found = False
        nx, ny = cx, cy
        # 左下
        if (h_open(maze, cx, cy) and v_open(maze, cx, cy)
                and h_open(maze, cx - 1, cy) and v_open(maze, cx, cy + 1)
                and cx > 0 and cy < h - 1 and value_fn(cx - 1, cy + 1) > best):
            best = value_fn(cx - 1, cy + 1)
            nx, ny = cx - 1, cy + 1
            found = True
        # 右下
        if (h_open(maze, cx, cy) and v_open(maze, cx + 1, cy)
                and h_open(maze, cx + 1, cy) and v_open(maze, cx + 1, cy + 1)
                and cx < w - 1 and cy < h - 1 and value_fn(cx + 1, cy + 1) > best):
            best = value_fn(cx + 1, cy + 1)
            nx, ny = cx + 1, cy + 1
            found = True
        # 左上
        if (v_open(maze, cx, cy) and h_open(maze, cx, cy - 1)
                and v_open(maze, cx, cy - 1) and h_open(maze, cx - 1, cy - 1)
                and cx > 0 and cy > 0 and value_fn(cx - 1, cy - 1) > best):
            best = value_fn(cx - 1, cy - 1)
            nx, ny = cx - 1, cy - 1
            found = True
        # 右上
        if (v_open(maze, cx + 1, cy) and h_open(maze, cx, cy - 1)
                and h_open(maze, cx + 1, cy - 1) and v_open(maze, cx + 1, cy - 1)
                and cx < w - 1 and cy > 0 and value_fn(cx + 1, cy - 1) > best):
            best = value_fn(cx + 1, cy - 1)
            nx, ny = cx + 1, cy - 1
            found = True
        # 左
        if v_open(maze, cx, cy) and cx > 0 and value_fn(cx - 1, cy) > best:
            best = value_fn(cx - 1, cy)
            nx, ny = cx - 1, cy
            found = True
        # 右
        if v_open(maze, cx + 1, cy) and cx < w - 1 and value_fn(cx + 1, cy) > best:
            best = value_fn(cx + 1, cy)
            nx, ny = cx + 1, cy
            found = True
        # 上
        if h_open(maze, cx, cy - 1) and cy > 0 and value_fn(cx, cy - 1) > best:
            best = value_fn(cx, cy - 1)
            nx, ny = cx, cy - 1
            found = True
        # 下
        if h_open(maze, cx, cy) and cy < h - 1 and value_fn(cx, cy + 1) > best:
            best = value_fn(cx, cy + 1)
            nx, ny = cx, cy + 1
            found = True

        cx, cy = nx, ny
        path.append({"x": cx, "y": cy})
        max_length -= 1
        if not (found and max_length > 0):
            break
    return path


def follow_gradient_path_with_distances(maze, distances, startx, starty, max_length):
    """frame_53:428-505"""
    return _gradient_walk(maze, lambda x, y: _d(distances, x, y),
                          startx, starty, max_length)


def follow_gradient_path_with_distances_and_dead_ends(
        maze, distances, dead_ends, startx, starty, max_length):
    """frame_53:506-586 — 值 = distances - deadEnds (undefined 参与运算得 NaN)"""
    def value(x, y):
        d = _d(distances, x, y)
        if 0 <= x < len(dead_ends) and 0 <= y < len(dead_ends[x]):
            de = dead_ends[x][y]
            de = NAN if de is None else de
        else:
            de = NAN
        return d - de
    return _gradient_walk(maze, value, startx, starty, max_length)


# ---------------------------------------------------------------- 墙壁几何

def build_wall_segments(maze, scale):
    """依据 drawMaze (frame_53:587-692) 生成碰撞用墙壁线段列表。

    - 坐标与原版一致: Math.floor(格线 * scale) 取整
    - 每格: [1]!=0 画下边墙, [2]!=0 画左边墙
    - 外边界四边恒有墙
    返回 [(x1,y1,x2,y2), ...] 全部为轴对齐线段。
    线条厚度 = 2*floor(scale/16), 方头端帽 (碰撞半厚 = floor(scale/16))。
    """
    w = len(maze)
    h = len(maze[0])
    fl = math.floor
    segs = []
    for x in range(w):
        for y in range(h):
            if maze[x][y][1] != 0:  # 下边墙
                segs.append((fl(x * scale), fl((y + 1) * scale),
                             fl((x + 1) * scale), fl((y + 1) * scale)))
            if maze[x][y][2] != 0:  # 左边墙
                segs.append((fl(x * scale), fl(y * scale),
                             fl(x * scale), fl((y + 1) * scale)))
    # 外边界 (maze[x][y][0] 恒为 1)
    for x in range(w):
        segs.append((fl(x * scale), 0, fl((x + 1) * scale), 0))
        segs.append((fl(x * scale), fl(h * scale), fl((x + 1) * scale), fl(h * scale)))
    for y in range(h):
        segs.append((0, fl((y + 1) * scale), 0, fl(y * scale)))
        segs.append((fl(w * scale), fl((y + 1) * scale), fl(w * scale), fl(y * scale)))
    return segs


def point_hits_walls(walls, half_t, px, py):
    """点是否命中墙壁线条 (mazemc.hitTest(x, y, true) 的等价实现)。

    线条厚度 2*half_t, 端帽为方头 (square caps): 两端各外延 half_t。
    所有墙均为轴对齐线段 → 命中即"点在线段外扩 half_t 的矩形内"。
    """
    for (x1, y1, x2, y2) in walls:
        if x1 == x2:  # 竖直
            if (abs(px - x1) <= half_t
                    and min(y1, y2) - half_t <= py <= max(y1, y2) + half_t):
                return True
        else:  # 水平
            if (abs(py - y1) <= half_t
                    and min(x1, x2) - half_t <= px <= max(x1, x2) + half_t):
                return True
    return False


class WallGrid:
    """墙壁碰撞的空间索引 — 结果与 point_hits_walls 完全一致, 仅提速。

    轴对齐线段 + 方头端帽的笔画区域 == 线段包围盒向四周外扩 half_t
    的矩形, 故点命中判定精确等价于点-矩形包含测试。
    """

    __slots__ = ("cell", "buckets")

    def __init__(self, walls, half_t, bucket_size=64.0):
        self.cell = bucket_size
        self.buckets = {}
        for (x1, y1, x2, y2) in walls:
            xmin = min(x1, x2) - half_t
            xmax = max(x1, x2) + half_t
            ymin = min(y1, y2) - half_t
            ymax = max(y1, y2) + half_t
            rect = (xmin, ymin, xmax, ymax)
            bx0 = int(xmin // bucket_size)
            bx1 = int(xmax // bucket_size)
            by0 = int(ymin // bucket_size)
            by1 = int(ymax // bucket_size)
            for bx in range(bx0, bx1 + 1):
                for by in range(by0, by1 + 1):
                    self.buckets.setdefault((bx, by), []).append(rect)

    def hit(self, px, py):
        rects = self.buckets.get((int(px // self.cell), int(py // self.cell)))
        if not rects:
            return False
        for (xmin, ymin, xmax, ymax) in rects:
            if xmin <= px <= xmax and ymin <= py <= ymax:
                return True
        return False
