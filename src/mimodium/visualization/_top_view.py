from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from matplotlib import path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.transforms import Affine2D
from scipy.spatial import ConvexHull, Voronoi

from .. import algorithms as alg
from .. import evaluation as evl
from .. import scenario as scn
from ._marker import ap_marker, cpu_marker, ue_marker


class _NodePatch(PathPatch, ABC):
    def __init__(self, x: float, y: float, marker: path.Path, size: float, color: str):
        self.x = x
        self.y = y
        self.pos = np.array([x, y])
        self.size = size
        self.marker = marker.transformed(Affine2D().scale(size))
        super().__init__(
            self.marker.transformed(Affine2D().translate(self.x, self.y)),
            lw=0.0,
            fc=color,
        )

    @abstractmethod
    def connection_points(self) -> np.ndarray:
        pass

    @abstractmethod
    def bounding_points(self, padding: float = 0.0) -> np.ndarray:
        pass


class UEPatch(_NodePatch):
    def __init__(self, x, y, size=50.0, color="black"):
        super().__init__(x, y, ue_marker, size, color)

    def connection_points(self):
        return np.vstack(
            (
                self.pos + self.size * np.array([0, 0.5]),
                self.pos + self.size * np.array([0, -0.5]),
            )
        )

    def bounding_points(self, padding: float = 0.7):
        # height: 1.0, width: 6/11
        return np.vstack(
            (
                self.pos + (1.0 + padding) * self.size * np.array([3 / 11, 0.5]),
                self.pos + (1.0 + padding) * self.size * np.array([-3 / 11, 0.5]),
                self.pos + (1.0 + padding) * self.size * np.array([3 / 11, -0.5]),
                self.pos + (1.0 + padding) * self.size * np.array([-3 / 11, -0.5]),
            )
        )


class APPatch(_NodePatch):
    def __init__(self, x, y, size=50.0, color="blue"):
        super().__init__(x, y, ap_marker, size, color)

    def connection_points(self):
        return np.vstack(
            (
                self.pos + 0.8 * self.size * np.array([0, 0.5]),
                self.pos + 0.8 * self.size * np.array([0, -0.5]),
            )
        )

    def bounding_points(self, padding: float = 0.7):
        # height: 1.0, width: 0.7469065300647367
        return np.vstack(
            (
                self.pos
                + (1.0 + padding) * self.size * np.array([0.37345326503236836, 0.5]),
                self.pos
                + (1.0 + padding) * self.size * np.array([-0.37345326503236836, 0.5]),
                self.pos
                + (1.0 + padding) * self.size * np.array([0.37345326503236836, -0.5]),
                self.pos
                + (1.0 + padding) * self.size * np.array([-0.37345326503236836, -0.5]),
            )
        )


class CPUPatch(_NodePatch):
    def __init__(self, x, y, size=50.0, color="red"):
        super().__init__(x, y, cpu_marker, size, color)

    def connection_points(self):
        return np.vstack(
            (
                self.pos + 1.05 * self.size * np.array([0.5, 0]),
                self.pos + 1.05 * self.size * np.array([-0.5, 0]),
                self.pos + 1.05 * self.size * np.array([0, 0.5]),
                self.pos + 1.05 * self.size * np.array([0, -0.5]),
            )
        )

    def bounding_points(self, padding: float = 0.7):
        padding /= 2
        # height: 1.0, width: 1.0
        return np.vstack(
            (
                self.pos + (1.0 + padding) * self.size * np.array([0.5, 0.5]),
                self.pos + (1.0 + padding) * self.size * np.array([-0.5, 0.5]),
                self.pos + (1.0 + padding) * self.size * np.array([0.5, -0.5]),
                self.pos + (1.0 + padding) * self.size * np.array([-0.5, -0.5]),
            )
        )


class CurvedLinePatch(PathPatch):
    def __init__(self, from_node: _NodePatch, to_node: _NodePatch, **kwargs):
        from_points = from_node.connection_points()
        to_points = to_node.connection_points()
        p1 = from_points[np.argmin(np.linalg.norm(from_points - to_node.pos, axis=1))]
        p4 = to_points[np.argmin(np.linalg.norm(to_points - from_node.pos, axis=1))]
        curve = 0.3 * np.linalg.norm(p1 - p4)
        p2 = p1 - curve * (from_node.pos - p1) / np.linalg.norm(from_node.pos - p1)
        p3 = p4 - curve * (to_node.pos - p4) / np.linalg.norm(to_node.pos - p4)
        super().__init__(
            path.Path(
                [p1, p2, p3, p4],
                [
                    path.Path.MOVETO,
                    path.Path.CURVE4,
                    path.Path.CURVE4,
                    path.Path.CURVE4,
                ],
            ),
            **kwargs,
            fc="none",
        )


class StraightLinePatch(PathPatch):
    def __init__(self, p1, p2, **kwargs):
        super().__init__(
            path.Path([p1, p2], [path.Path.MOVETO, path.Path.LINETO]),
            **kwargs,
            fc="none",
        )


class MidhaulPatch(CurvedLinePatch):
    def __init__(self, from_cpu: _NodePatch, to_cpu: _NodePatch, **kwargs):
        super().__init__(from_cpu, to_cpu, **kwargs)


class FronthaulPatch(CurvedLinePatch):
    def __init__(self, from_ap: _NodePatch, to_cpu: _NodePatch, **kwargs):
        super().__init__(from_ap, to_cpu, **kwargs)


class WirelessPatch(CurvedLinePatch):
    def __init__(self, from_ue: _NodePatch, to_ap: _NodePatch, **kwargs):
        super().__init__(from_ue, to_ap, **kwargs)


class ClusterPatch(PathPatch):
    # TODO: exclude APs that are inside the convex shape, but not used by the UE
    def __init__(
        self,
        included: Sequence[_NodePatch],
        roundness: float = 0.3,
        padding: float = 0.7,
        **kwargs,
    ):
        # maybe add pilot ID to each UE by filling its "screen"? or a number?
        points = np.vstack([n.bounding_points(padding) for n in included])
        hull_points = points[ConvexHull(points, incremental=True).vertices]
        vertices = []
        codes = []
        for i, point in enumerate(hull_points):
            next_p = hull_points[(i + 1) % len(hull_points)]
            prev_p = hull_points[i - 1]
            ref_direction = (next_p - prev_p) / np.linalg.norm(next_p - prev_p)
            next_dist = np.linalg.norm(next_p - point)
            prev_dist = np.linalg.norm(prev_p - point)
            ref_pre = point - roundness * ref_direction * prev_dist
            ref_post = point + roundness * ref_direction * next_dist
            vertices.append(ref_pre)
            vertices.append(point)
            vertices.append(ref_post)
            codes.append(path.Path.CURVE4)
            codes.append(path.Path.CURVE4)
            codes.append(path.Path.CURVE4)
        vertices = vertices[1:] + vertices[:2]
        codes[0] = path.Path.MOVETO
        codes.append(path.Path.CURVE4)
        super().__init__(path.Path(vertices, codes), **kwargs)


def add_heatmap_of_ue_ses(
    ax: Axes,
    res: list[
        tuple[
            scn.UEPositions,
            evl.spectral_efficiency.UplinkSpectralEfficiencies
            | evl.spectral_efficiency.DownlinkSpectralEfficiencies,
        ]
    ],
    num_bins: int,
    bin_size: float,
    cmap: ScalarMappable,
    percentile: float = 50,
):
    ue_pos_se_pairs: list[tuple] = []
    for r in res:
        ue_pos = r[0]
        ue_se = r[1]
        ue_pos_se_pairs.extend(list(zip(ue_pos, ue_se)))

    xy_with_se_lists = [[[] for _ in range(num_bins)] for _ in range(num_bins)]
    for ue_pos, ue_se in ue_pos_se_pairs:
        x_bin = int(ue_pos[0] // bin_size)
        y_bin = int(ue_pos[1] // bin_size)
        xy_with_se_lists[x_bin][y_bin].append(ue_se)

    for xi, x_list in enumerate(xy_with_se_lists):
        for yi, y_list in enumerate(x_list):
            if y_list:
                se_value = np.percentile(y_list, percentile)
                ax.add_patch(
                    Rectangle(
                        (xi * bin_size, yi * bin_size),
                        bin_size,
                        bin_size,
                        color=cmap.to_rgba(se_value),  # type: ignore
                        alpha=0.7,
                    )
                )


def make_ap_patches(positions: scn.APPositions, size: float = 50) -> list[APPatch]:
    return [APPatch(x, y, size=size) for x, y in positions]


def make_ue_patches(positions: scn.UEPositions, size: float = 50) -> list[UEPatch]:
    return [UEPatch(x, y, size=size) for x, y in positions]


def make_cpu_patches(
    positions: scn.cpu.CPUPositions, size: float = 50
) -> list[CPUPatch]:
    return [CPUPatch(x, y, size=size) for x, y in positions]


def make_fronthaul_patches(
    ap_patches: Sequence[APPatch],
    cpu_patches: Sequence[CPUPatch],
    fronthaul: scn.FronthaulLinks,
    linewidth: float = 2.0,
    linestyle: str = "solid",
    color: str = "blue",
) -> list[FronthaulPatch]:
    assert len(ap_patches) == fronthaul.shape[0]
    assert len(cpu_patches) == fronthaul.shape[1]
    fronthaul_patches: list[FronthaulPatch] = []
    for l, ap_patch in enumerate(ap_patches):
        for j, cpu_patch in enumerate(cpu_patches):
            if fronthaul[l, j]:
                fronthaul_patches.append(
                    FronthaulPatch(
                        ap_patch,
                        cpu_patch,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        ec=color,
                    )
                )
    return fronthaul_patches


def make_midhaul_patches(
    mid_links: scn.MidhaulLinks,
    cpu_patches: Sequence[CPUPatch],
    linewidth: float = 2.0,
    linestyle: str = "solid",
    color: str = "red",
) -> list[MidhaulPatch]:
    assert len(cpu_patches) == mid_links.shape[0]
    midhaul_patches: list[MidhaulPatch] = []
    for j1, cpu1 in enumerate(cpu_patches):
        for j2, cpu2 in enumerate(cpu_patches[j1 + 1 :]):
            if mid_links[j1, j2 + j1 + 1]:
                midhaul_patches.append(
                    MidhaulPatch(
                        cpu1, cpu2, linewidth=linewidth, linestyle=linestyle, ec=color
                    )
                )
    return midhaul_patches


def make_wireless_patches(
    ue_patches: Sequence[UEPatch],
    ap_patches: Sequence[APPatch],
    used_links: alg.UsedWirelessLinks,
    linewidth: float = 1.0,
    linestyle: str = "dashed",
    color: str = "0.65",
) -> list[WirelessPatch]:
    """Create one wireless-link patch for every active UE-to-AP link."""
    assert len(ue_patches) == used_links.shape[0]
    assert len(ap_patches) == used_links.shape[1]
    wireless_patches: list[WirelessPatch] = [
        WirelessPatch(
            ue_patch,
            ap_patch,
            linewidth=linewidth,
            linestyle=linestyle,
            ec=color,
        )
        for k, ue_patch in enumerate(ue_patches)
        for l, ap_patch in enumerate(ap_patches)
        if used_links[k, l]
    ]
    return wireless_patches


def make_cell_edge_patches(
    cpu_positions: scn.CPUPositions,
    linestyle: str = "dotted",
    linewidth: float = 2.0,
    color="gray",
) -> list[StraightLinePatch]:
    vor = Voronoi(cpu_positions)
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)
    verts = vor.vertices
    cell_edges: list[StraightLinePatch] = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            cell_edge = StraightLinePatch(
                verts[simplex[0]],
                verts[simplex[1]],
                ec=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if vor.furthest_site:
                direction = -direction
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

            cell_edge = StraightLinePatch(
                verts[i],
                far_point,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        cell_edges.append(cell_edge)
    return cell_edges


def make_service_cluster_patches(
    ap_patches: Sequence[APPatch],
    ue_patches: Sequence[UEPatch],
    used_links: alg.UsedWirelessLinks,
    pilot_ids: alg.AssignedPilotIDs,
) -> list[ClusterPatch]:
    cluster_patches: list[ClusterPatch] = []
    for ue_patch, used_aps, pilot_id in zip(ue_patches, used_links, pilot_ids):
        serving_aps = list(np.array(ap_patches)[used_aps])
        cluster_patches.append(
            ClusterPatch(
                [ue_patch] + serving_aps,
                padding=0.1,
                fill=True,
                color=f"C{pilot_id + 1}",
                alpha=0.8,
                linestyle="solid",
            )
        )
    return cluster_patches


def make_estimated_cluster_patches(
    ap_patches: Sequence[APPatch],
    ue_patches: Sequence[UEPatch],
    estimated_links: alg.EstimatedChannelLinks,
    pilot_ids: alg.AssignedPilotIDs,
) -> list[ClusterPatch]:
    cluster_patches: list[ClusterPatch] = []
    for ue_patch, used_aps, pilot_id in zip(ue_patches, estimated_links, pilot_ids):
        serving_aps = list(np.array(ap_patches)[used_aps])
        cluster_patches.append(
            ClusterPatch(
                [ue_patch] + serving_aps,
                padding=0.5,
                color=f"C{pilot_id + 1}",
                alpha=0.5,
                fill=True,
                linestyle="dashed",
            )
        )
    return cluster_patches


def make_measuring_cluster_patches(
    ap_patches: Sequence[APPatch],
    ue_patches: Sequence[UEPatch],
    measured_links: alg.MeasuredStatisticLinks,
    pilot_ids: alg.AssignedPilotIDs,
) -> list[ClusterPatch]:
    cluster_patches: list[ClusterPatch] = []
    for ue_patch, used_aps, pilot_id in zip(ue_patches, measured_links, pilot_ids):
        serving_aps = list(np.array(ap_patches)[used_aps])
        cluster_patches.append(
            ClusterPatch(
                [ue_patch] + serving_aps,
                padding=1.0,
                color=f"C{pilot_id + 1}",
                alpha=0.4,
                fill=True,
                linestyle="dotted",
            )
        )
    return cluster_patches


def plot_top_view(
    ap_positions: scn.APPositions,
    ue_positions: scn.UEPositions,
    *,
    cpu_positions: scn.CPUPositions | None = None,
    fronthaul_links: scn.FronthaulLinks | None = None,
    midhaul_links: scn.MidhaulLinks | None = None,
    used_links: alg.UsedWirelessLinks | None = None,
    estimated_links: alg.EstimatedChannelLinks | None = None,
    measured_links: alg.MeasuredStatisticLinks | None = None,
    pilot_ids: alg.AssignedPilotIDs | None = None,
    area_length: scn.AreaLength | None = None,
    node_size: float = 50.0,
    cell_edges: bool = False,
    padding: float = 0.05,
    axis_off: bool = True,
    grid_on: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot a top view directly from scenario and algorithm result arrays.

    Optional link matrices add their corresponding connections. Supplying pilot
    IDs adds service, estimated, and measured clusters for each supplied link
    matrix. Pass an axes to compose with an existing figure, or omit it to create
    a new one.
    """
    if cpu_positions is None and (
        fronthaul_links is not None or midhaul_links is not None or cell_edges
    ):
        raise ValueError(
            "cpu_positions are required for fronthaul, midhaul, and cell edges"
        )
    if pilot_ids is None and (
        estimated_links is not None or measured_links is not None
    ):
        raise ValueError("pilot_ids are required for estimated and measured clusters")

    if ax is None:
        _, ax = plt.subplots()

    ap_patches = make_ap_patches(ap_positions, size=node_size)
    ue_patches = make_ue_patches(ue_positions, size=node_size)
    cpu_patches = (
        make_cpu_patches(cpu_positions, size=node_size)
        if cpu_positions is not None
        else []
    )

    plot_patches: list[PathPatch] = []
    if pilot_ids is not None:
        if measured_links is not None:
            plot_patches.extend(
                make_measuring_cluster_patches(
                    ap_patches, ue_patches, measured_links, pilot_ids
                )
            )
        if estimated_links is not None:
            plot_patches.extend(
                make_estimated_cluster_patches(
                    ap_patches, ue_patches, estimated_links, pilot_ids
                )
            )
        if used_links is not None:
            plot_patches.extend(
                make_service_cluster_patches(
                    ap_patches, ue_patches, used_links, pilot_ids
                )
            )
    if cell_edges:
        assert cpu_positions is not None
        plot_patches.extend(make_cell_edge_patches(cpu_positions))
    if midhaul_links is not None:
        plot_patches.extend(make_midhaul_patches(midhaul_links, cpu_patches))
    if fronthaul_links is not None:
        plot_patches.extend(
            make_fronthaul_patches(ap_patches, cpu_patches, fronthaul_links)
        )
    if used_links is not None:
        plot_patches.extend(make_wireless_patches(ue_patches, ap_patches, used_links))
    plot_patches.extend(ap_patches + ue_patches + cpu_patches)
    [ax.add_patch(plot_patch) for plot_patch in plot_patches]

    return get_square_axes(
        area_length,
        padding=padding,
        axis_off=axis_off,
        grid_on=grid_on,
        ax=ax,
    )


def get_square_axes(
    area_length: scn.AreaLength | None = None,
    padding: float = 0.1,
    axis_off: bool = True,
    grid_on: bool = False,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1)
    assert isinstance(ax, Axes)

    if area_length is None:
        ax.autoscale_view()
        ax.margins(padding)
    else:
        ax.set_xlim(-padding * area_length, area_length * (1 + padding))
        ax.set_ylim(-padding * area_length, area_length * (1 + padding))

    ax.grid(visible=grid_on)
    ax.set_aspect("equal", adjustable="box")
    if axis_off:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
    return ax


def set_exact_axes_size(w: float, h: float, ax: Axes):
    """w, h: width, height in inches"""
    l = ax.figure.subplotpars.left  # type: ignore
    r = ax.figure.subplotpars.right  # type: ignore
    t = ax.figure.subplotpars.top  # type: ignore
    b = ax.figure.subplotpars.bottom  # type: ignore
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)  # type: ignore
