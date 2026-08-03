from pathlib import Path

import matplotlib.pyplot as plt

from ._marker import ap_marker, ap_thin_marker, cpu_marker, ue_marker
from ._top_view import (
    make_ap_patches,
    make_cell_edge_patches,
    make_cpu_patches,
    make_estimated_cluster_patches,
    make_fronthaul_patches,
    make_measuring_cluster_patches,
    make_midhaul_patches,
    make_service_cluster_patches,
    make_ue_patches,
    make_wireless_patches,
    plot_top_view,
)

_STYLE_DIR = Path(__file__).parent
_DEFAULT_STYLE = _STYLE_DIR / "mimodium.mplstyle"


def use_default_style() -> None:
    """Apply Mimodium's default Matplotlib style."""
    plt.style.use(_DEFAULT_STYLE)


__all__ = (
    "ap_marker",
    "ap_thin_marker",
    "cpu_marker",
    "make_ap_patches",
    "make_cell_edge_patches",
    "make_cpu_patches",
    "make_estimated_cluster_patches",
    "make_fronthaul_patches",
    "make_measuring_cluster_patches",
    "make_midhaul_patches",
    "make_service_cluster_patches",
    "make_ue_patches",
    "make_wireless_patches",
    "plot_top_view",
    "ue_marker",
    "use_default_style",
)
