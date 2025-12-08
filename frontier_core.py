from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Protocol, Optional, Any

import numpy as np


# ----------------------------------------------------------------------
# Costmap interface
# ----------------------------------------------------------------------


class CostmapProtocol(Protocol):
    """Minimal interface of a 2D costmap required for frontier search.

    Any object passed here must implement these methods. This is
    intentionally close to nav2_costmap_2d::Costmap2D.
    """

    def getSizeInCellsX(self) -> int: ...
    def getSizeInCellsY(self) -> int: ...
    def getResolution(self) -> float: ...
    def getCharMap(self) -> Sequence[int]: ...
    def mapToWorld(self, mx: int, my: int) -> Tuple[float, float]: ...
    def worldToMap(self, wx: float, wy: float) -> Tuple[bool, int, int]: ...
    def indexToCells(self, index: int) -> Tuple[int, int]: ...


# ----------------------------------------------------------------------
# Simple geometry / frontier structures
# ----------------------------------------------------------------------


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Frontier:
    """Cluster of frontier cells."""
    size: int = 0
    min_distance: float = 0.0
    cost: float = 0.0
    initial: Point = field(default_factory=Point)
    centroid: Point = field(default_factory=Point)
    middle: Point = field(default_factory=Point)
    points: List[Point] = field(default_factory=list)


# nav2_costmap_2d semantics (unsigned char map values)
FREE_SPACE: int = 0
NO_INFORMATION: int = 255


# ----------------------------------------------------------------------
# Helper functions (ported from costmap_tools.h)
# ----------------------------------------------------------------------


def _nhood4(idx: int, costmap: CostmapProtocol) -> List[int]:
    """4-connected neighbours of a cell index inside the map."""
    size_x = costmap.getSizeInCellsX()
    size_y = costmap.getSizeInCellsY()
    out: List[int] = []

    if idx < 0 or idx >= size_x * size_y:
        return out

    # left
    if idx % size_x > 0:
        out.append(idx - 1)
    # right
    if idx % size_x < size_x - 1:
        out.append(idx + 1)
    # up
    if idx >= size_x:
        out.append(idx - size_x)
    # down
    if idx < size_x * (size_y - 1):
        out.append(idx + size_x)

    return out


def _nhood8(idx: int, costmap: CostmapProtocol) -> List[int]:
    """8-connected neighbours of a cell index inside the map."""
    size_x = costmap.getSizeInCellsX()
    size_y = costmap.getSizeInCellsY()
    out = _nhood4(idx, costmap)

    if idx < 0 or idx >= size_x * size_y:
        return out

    # upper-left
    if idx % size_x > 0 and idx >= size_x:
        out.append(idx - 1 - size_x)
    # lower-left
    if idx % size_x > 0 and idx < size_x * (size_y - 1):
        out.append(idx - 1 + size_x)
    # upper-right
    if idx % size_x < size_x - 1 and idx >= size_x:
        out.append(idx + 1 - size_x)
    # lower-right
    if idx % size_x < size_x - 1 and idx < size_x * (size_y - 1):
        out.append(idx + 1 + size_x)

    return out


def _nearest_cell(start: int, value: int, costmap: CostmapProtocol) -> Optional[int]:
    """Breadth-first search for the nearest cell with a given value."""
    size_x = costmap.getSizeInCellsX()
    size_y = costmap.getSizeInCellsY()
    total = size_x * size_y

    if start < 0 or start >= total:
        return None

    cmap = np.asarray(costmap.getCharMap(), dtype=np.uint8).reshape(-1)
    visited = np.zeros(total, dtype=bool)

    from collections import deque

    q = deque()
    q.append(start)
    visited[start] = True

    target_val = np.uint8(value)

    while q:
        idx = q.popleft()
        if cmap[idx] == target_val:
            return idx

        for nbr in _nhood8(idx, costmap):
            if not visited[nbr]:
                visited[nbr] = True
                q.append(nbr)

    return None


# ----------------------------------------------------------------------
# Frontier search core
# ----------------------------------------------------------------------


class FrontierSearch:
    """Frontier detection / clustering on a 2D costmap.

    C++ FrontierSearch(frontier_exploration / explore_lite)의
    핵심 로직만 파이썬으로 옮긴 버전.
    ROS Node / Action 같은 의존성은 모두 제거했다.
    """

    def __init__(
        self,
        costmap: CostmapProtocol,
        potential_scale: float = 1e-3,
        gain_scale: float = 1.0,
        min_frontier_size: float = 0.5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._costmap = costmap
        self._map: np.ndarray | None = None
        self._size_x: int = 0
        self._size_y: int = 0
        self._potential_scale = float(potential_scale)
        self._gain_scale = float(gain_scale)
        self._min_frontier_size = float(min_frontier_size)
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_from(self, position: Any) -> List[Frontier]:
        """Compute all frontiers reachable from the given world position.

        Args:
            position:
              - frontier_core.Point, 또는
              - .x, .y (선택적으로 .z) 필드를 가진 객체
                (예: geometry_msgs.msg.Point, geometry_msgs.msg.Pose.position)

        Returns:
            Frontier 리스트 (cost 오름차순 정렬)
        """
        pos = self._to_point(position)

        # world -> map coordinates
        ok, mx, my = self._costmap.worldToMap(pos.x, pos.y)
        if not ok:
            self._logger.error(
                "Robot pose (%.3f, %.3f) is out of costmap bounds",
                pos.x,
                pos.y,
            )
            return []

        # cache map + sizes
        cmap = np.asarray(self._costmap.getCharMap(), dtype=np.uint8).reshape(-1)
        self._map = cmap
        self._size_x = self._costmap.getSizeInCellsX()
        self._size_y = self._costmap.getSizeInCellsY()

        total_cells = self._size_x * self._size_y
        frontier_flag = np.zeros(total_cells, dtype=bool)
        visited_flag = np.zeros(total_cells, dtype=bool)

        from collections import deque

        if hasattr(self._costmap, "getIndex"):
            start_index = self._costmap.getIndex(mx, my)  # type: ignore[attr-defined]
        else:
            start_index = my * self._size_x + mx

        bfs = deque()
        clear = _nearest_cell(start_index, FREE_SPACE, self._costmap)
        if clear is not None:
            bfs.append(clear)
        else:
            bfs.append(start_index)
            self._logger.warning(
                "Could not find nearby clear cell; starting frontier search from robot cell"
            )

        visited_flag[bfs[0]] = True
        frontiers: List[Frontier] = []

        while bfs:
            idx = bfs.popleft()

            for nbr in _nhood4(idx, self._costmap):
                # free space wavefront
                if (
                    self._map[nbr] <= self._map[idx]
                    and not visited_flag[nbr]
                ):
                    visited_flag[nbr] = True
                    bfs.append(nbr)
                else:
                    # frontier cell candidate
                    if self._is_new_frontier_cell(nbr, frontier_flag):
                        frontier_flag[nbr] = True
                        new_frontier = self._build_new_frontier(
                            initial_cell=nbr,
                            reference=start_index,
                            frontier_flag=frontier_flag,
                        )
                        # 너무 작은 프론티어 필터링 (단위: m)
                        if (
                            new_frontier.size * self._costmap.getResolution()
                            >= self._min_frontier_size
                        ):
                            frontiers.append(new_frontier)

        # cost 계산 + 정렬
        for fr in frontiers:
            fr.cost = self._frontier_cost(fr)

        frontiers.sort(key=lambda f: f.cost)
        return frontiers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_point(self, obj: Any) -> Point:
        """Convert arbitrary object with x, y, (optional z) into Point."""
        if isinstance(obj, Point):
            return obj
        if hasattr(obj, "x") and hasattr(obj, "y"):
            z = getattr(obj, "z", 0.0)
            return Point(float(obj.x), float(obj.y), float(z))
        raise TypeError(
            f"position must be frontier_core.Point or have .x/.y attributes, got {type(obj)!r}"
        )

    def _is_new_frontier_cell(
        self, idx: int, frontier_flag: np.ndarray
    ) -> bool:
        """Return True if cell is a new frontier cell."""
        assert self._map is not None

        # unknown 이고, 아직 frontier로 안 찍힌 셀이어야 함
        if self._map[idx] != NO_INFORMATION or frontier_flag[idx]:
            return False

        # 4-이웃 중 free space가 하나라도 있으면 frontier
        for nbr in _nhood4(idx, self._costmap):
            if self._map[nbr] == FREE_SPACE:
                return True

        return False

    def _build_new_frontier(
        self,
        initial_cell: int,
        reference: int,
        frontier_flag: np.ndarray,
    ) -> Frontier:
        """Flood-fill on 8-connected neighbourhood to build a frontier cluster."""
        output = Frontier()
        output.size = 1
        output.min_distance = math.inf
        output.cost = 0.0

        # initial cell info
        init_mx, init_my = self._costmap.indexToCells(initial_cell)
        init_wx, init_wy = self._costmap.mapToWorld(init_mx, init_my)
        output.initial = Point(x=init_wx, y=init_wy, z=0.0)

        # centroid accumulators (나중에 /size 해줌)
        output.centroid = Point(x=0.0, y=0.0, z=0.0)
        output.middle = Point(x=init_wx, y=init_wy, z=0.0)

        from collections import deque

        bfs = deque()
        bfs.append(initial_cell)

        # reference world position (robot cell) for distance
        ref_mx, ref_my = self._costmap.indexToCells(reference)
        ref_wx, ref_wy = self._costmap.mapToWorld(ref_mx, ref_my)

        while bfs:
            idx = bfs.popleft()

            for nbr in _nhood8(idx, self._costmap):
                if self._is_new_frontier_cell(nbr, frontier_flag):
                    frontier_flag[nbr] = True

                    mx, my = self._costmap.indexToCells(nbr)
                    wx, wy = self._costmap.mapToWorld(mx, my)

                    p = Point(x=wx, y=wy, z=0.0)
                    output.points.append(p)

                    # centroid 누적
                    output.centroid.x += wx
                    output.centroid.y += wy

                    output.size += 1

                    # reference(로봇 위치)에 가장 가까운 셀 = middle
                    dx = ref_wx - wx
                    dy = ref_wy - wy
                    dist = math.hypot(dx, dy)
                    if dist < output.min_distance:
                        output.min_distance = dist
                        output.middle = Point(x=wx, y=wy, z=0.0)

                    bfs.append(nbr)

        # centroid 평균
        if output.size > 0:
            output.centroid.x /= float(output.size)
            output.centroid.y /= float(output.size)

        return output

    def _frontier_cost(self, frontier: Frontier) -> float:
        """Same heuristic as original explore_lite:

        cost = potential_scale * distance - gain_scale * size
        (all in meters).
        """
        return (
            self._potential_scale * frontier.min_distance * self._costmap.getResolution()
            - self._gain_scale * frontier.size * self._costmap.getResolution()
        )
