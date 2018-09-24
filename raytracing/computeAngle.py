"""
This module contains functions to compute the angle at which a seismic ray
must leave the tremor source at the plate boundary to arrive at the seismic
station at the surface
"""

from math import sqrt, pi, asin
from numpy import arange

from data import VsUOC, VpUOC, VsLOC, VpLOC, VsCC, VpCC

from computeDistance import computeDistance3, computeDistance5
from misc import computeDip

def computeAngle3(x0, y0, d, x, y, R1='S', R2='S', R3='S'):
    """
    Test different angles of incidence of ray departing from the source
    and find the one corresponding to station-source distance
    (one reflection at the mid-slab discontinuity)

    Input:
        type x0 = float
        x0 = EW coordinate of the source (in m)
        type y0 = float
        y0 = NS coordinate of the source (in m)
        type d = float
        d = Depth of the source (in km)
        type x = float
        x = EW coordinate of the station (in m)
        type y = float
        y = NS coordinate of the station (in m)
        type R1 = string
        R1 = Type of the downgoing ray in UOC (S or P)
        type R2 = string
        R2 = Type of the upgoing ray in UOC (S or P)
        type R3 = string
        R3 = Type of the upgoing ray in CC (S or P)
    Output:
        type angle = float
        angle = Best-fitting angle (in degrees with precision 0.01)
    """
    # Checking input data
    assert (R1 == 'S' or R1 == 'P'), "Ray 1 must be a P or an S wave!"
    assert (R2 == 'S' or R2 == 'P'), "Ray 2 must be a P or an S wave!"
    assert (R3 == 'S' or R3 == 'P'), "Ray 3 must be a P or an S wave!"
    # Compute distance between source and receiver
    distance = sqrt((x0 - x) ** 2.0 + (y0 - y) ** 2.0)
    # Compute dipping in the source-receiver vertical plane
    dip = computeDip(x0, y0, x, y)
    # Assign velocities
    if R1 == 'S':
        V1 = VsUOC
    else:
        V1 = VpUOC
    if R2 == 'S':
        V2 = VsUOC
    else:
        V2 = VpUOC
    if R3 == 'S':
        V3 = VsCC
    else:
        V3 = VpCC
    # Maximum angle for transmitted / reflected wave
    if V1 > V2:
        max1 = 90.0 + dip
    else:
        max1 = dip + asin(V1 / V2) * 180.0 / pi
    if V1 > V3:
        max2 = 90.0 + dip
    else:
        max2 = dip + asin(V1 / V3) * 180.0 / pi
    # Loop 1: Find angle with one degree precision
    minAngle = dip
    maxAngle = min(max1, max2, 90.0 + dip)
    stepAngle = 1.0
    angle = minAngle
    eps = abs(distance - computeDistance3(minAngle, x0, y0, d, x, y, R1, \
        R2, R3))
    for i in arange(minAngle, maxAngle, stepAngle):
        X = computeDistance3(i, x0, y0, d, x, y, R1, R2, R3)
        if (abs(distance - X) < eps):
            angle = i
            eps = abs(distance - X)
    # Loop 2: Find angle with 0.1 degree precision
    minAngle = angle - 1.0
    maxAngle = min(max1, max2, angle + 1.0)
    stepAngle = 0.1
    angle = minAngle
    eps = abs(distance - computeDistance3(minAngle, x0, y0, d, x, y, R1, \
        R2, R3))
    for i in arange(minAngle, maxAngle, stepAngle):
        X = computeDistance3(i, x0, y0, d, x, y, R1, R2, R3)
        if (abs(distance - X) < eps):
            angle = i
            eps = abs(distance - X)
    # Loop 3: Find angle with 0.01 degree precision
    minAngle = angle - 0.1
    maxAngle = min(max1, max2, angle + 0.1)
    stepAngle = 0.01
    angle = minAngle
    eps = abs(distance - computeDistance3(minAngle, x0, y0, d, x, y, R1, \
        R2, R3))
    for i in arange(minAngle, maxAngle, stepAngle):
        X = computeDistance3(i, x0, y0, d, x, y, R1, R2, R3)
        if (abs(distance - X) < eps):
            angle = i
            eps = abs(distance - X)
    return angle

def computeAngle5(x0, y0, d, x, y, R1='S', R2='S', R3='S', R4='S', R5='S'):
    """
    Test different angles of incidence of ray departing from the source
    and find the one corresponding to station-source distance
    (one reflection at the Moho)

    Input:
        type x0 = float
        x0 = EW coordinate of the source (in m)
        type y0 = float
        y0 = NS coordinate of the source (in m)
        type d = float
        d = Depth of the source (in km)
        type x = float
        x = EW coordinate of the station (in m)
        type y = float
        y = NS coordinate of the station (in m)
        type R1 = string
        R1 = Type of the downgoing ray in UOC (S or P)
        type R2 = string
        R2 = Type of the downgoing ray in LOC (S or P)
        type R3 = string
        R3 = Type of the upgoing ray in LOC (S or P)
        type R4 = string
        R4 = Type of the upgoing ray in UOC (S or P)
        type R5 = string
        R5 = Type of the upgoing ray in CC (S or P)
    Output:
        type angle = float
        angle = Best-fitting angle (in degrees with precision 0.01)
    """
    # Checking input data
    assert (R1 == 'S' or R1 == 'P'), "Ray 1 must be a P or an S wave!"
    assert (R2 == 'S' or R2 == 'P'), "Ray 2 must be a P or an S wave!"
    assert (R3 == 'S' or R3 == 'P'), "Ray 3 must be a P or an S wave!"
    assert (R4 == 'S' or R4 == 'P'), "Ray 4 must be a P or an S wave!"
    assert (R5 == 'S' or R5 == 'P'), "Ray 5 must be a P or an S wave!"
    # Compute distance between source and receiver
    distance = sqrt((x0 - x) ** 2.0 + (y0 - y) ** 2.0)
    # Compute dipping in the source-receiver vertical plane
    dip = computeDip(x0, y0, x, y)
    # Assign velocities
    if R1 == 'S':
        V1 = VsUOC
    else:
        V1 = VpUOC
    if R2 == 'S':
        V2 = VsLOC
    else:
        V2 = VpLOC
    if R3 == 'S':
        V3 = VsLOC
    else:
        V3 = VpLOC
    if R4 == 'S':
        V4 = VsUOC
    else:
        V4 = VpUOC
    if R5 == 'S':
        V5 = VsCC
    else:
        V5 = VpCC
    # Maximum angle for transmitted / reflected wave
    if V1 > V2:
        max1 = 90.0 + dip
    else:
        max1 = dip + asin(V1 / V2) * 180.0 / pi
    if V1 > V3:
        max2 = 90.0 + dip
    else:
        max2 = dip + asin(V1 / V3) * 180.0 / pi
    if V1 > V4:
        max3 = 90.0 + dip
    else:
        max3 = dip + asin(V1 / V4) * 180.0 / pi
    if V1 > V5:
        max4 = 90.0 + dip
    else:
        max4 = dip + asin(V1 / V5) * 180.0 / pi
    # Loop 1: Find angle with one degree precision
    minAngle = dip
    maxAngle = min(max1, max2, max3, max4, 90.0 + dip)
    stepAngle = 1.0
    angle = minAngle
    eps = abs(distance - computeDistance5(minAngle, x0, y0, d, x, y, R1, \
        R2, R3, R4, R5))
    for i in arange(minAngle, maxAngle, stepAngle):
        X = computeDistance5(i, x0, y0, d, x, y, R1, R2, R3, R4, R5)
        if (abs(distance - X) < eps):
            angle = i
            eps = abs(distance - X)
    # Loop 2: Find angle with 0.1 degree precision
    minAngle = angle - 1.0
    maxAngle = min(max1, max2, max3, max4, angle + 1.0)
    stepAngle = 0.1
    angle = minAngle
    eps = abs(distance - computeDistance5(minAngle, x0, y0, d, x, y, R1, \
        R2, R3, R4, R5))
    for i in arange(minAngle, maxAngle, stepAngle):
        X = computeDistance5(i, x0, y0, d, x, y, R1, R2, R3, R4, R5)
        if (abs(distance - X) < eps):
            angle = i
            eps = abs(distance - X)
    # Loop 3: Find angle with 0.01 degree precision
    minAngle = angle - 0.1
    maxAngle = min(max1, max2, max3, max4, angle + 0.1)
    stepAngle = 0.01
    angle = minAngle
    eps = abs(distance - computeDistance5(minAngle, x0, y0, d, x, y, R1, \
        R2, R3, R4, R5))
    for i in arange(minAngle, maxAngle, stepAngle):
        X = computeDistance5(i, x0, y0, d, x, y, R1, R2, R3, R4, R5)
        if (abs(distance - X) < eps):
            angle = i
            eps = abs(distance - X)
    return angle
