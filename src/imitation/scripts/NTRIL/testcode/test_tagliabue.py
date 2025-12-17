"""Replicate Tagliabue's method for Data Augmentation"""

from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
import scipy

import do_mpc
from scipy.linalg import solve_discrete_are
from scipy.spatial import ConvexHull
import pytope

from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC


def main():

    ## Step 1: Run expert MPC solver to generate expert demonstration under training distribution

    ## Step 2: Set up Robust Tube MPC to account for expected test distribution

    ## Step 3: Send expert demonstration as reference trajectory for RTMPC and return optimal solution

    ## Step 4: Generate samples at each timestep of optimal solution and solve for control using ancillary controller

    ## Step 5: Augment training database with samples, set up BC

    ## Step 6: Train BC policy

    return


if __name__ == "__main__":
    main()
