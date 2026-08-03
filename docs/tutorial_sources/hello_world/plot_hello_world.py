"""
Hello World
===========

This tutorial creates a small cell-free massive MIMO scenario with four access
points (APs) and three user equipments (UEs). It introduces the central
Mimodium pattern: configure a Dagreon workflow and request the result arrays
needed for a simulation or plot.
"""

import matplotlib.pyplot as plt
import mimodium as mimo
import mimodium.scenario as scn
import mimodium.visualization as vis
import numpy as np
from dagreon import Workflow

# %%
# Configure the scenario
# ----------------------
#
# ``core_tasks`` provides Mimodium's standard derived computations. The other
# tasks set the square area and the AP and UE positions used in this example.

workflow = Workflow(
    [
        *mimo.core_tasks(),
        scn.area.CfgAreaLength(1_000.0),
        scn.ap.CfgExplicitNumAPs(4),
        scn.ap.CfgEvenlySpacedAPPositions(),
        scn.ue.CfgExplicitUEPositions(
            np.array(
                [
                    [250.0, 500.0],
                    [500.0, 500.0],
                    [750.0, 500.0],
                ]
            )
        ),
    ]
)

# %%
# Request results
# ---------------
#
# Dagreon evaluates only the tasks needed to produce the requested result.

ap_positions = workflow.run(scn.APPositions)
ue_positions = workflow.run(scn.UEPositions)

print(f"AP positions:\n{ap_positions}")
print(f"UE positions:\n{ue_positions}")

# %%
# Plot the scenario
# -----------------
#
# Visualization helpers consume the same result arrays returned by the
# workflow.

ax = vis.plot_top_view(
    ap_positions,
    ue_positions,
    area_length=1_000.0,
    node_size=60.0,
    axis_off=False,
    grid_on=True,
)
ax.set_title("Hello, Mimodium!")
plt.show()
