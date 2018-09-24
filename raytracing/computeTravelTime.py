"""
This module contains functions to compute the seismic wave travel time
between the tremor source at the plate boundary and the seismic station
at the surface
"""

from math import pi, cos, sin, tan, asin

from data import VsUOC, VpUOC, VsLOC, VpLOC, VsCC, VpCC, hUOC, hLOC

from misc import computeDip

def computeTravelTime3(alpha, x0, y0, d, x, y, R1='S', R2='S', R3='S'):
    """
    Compute the travel time between the source and the receiver
    with one reflection at the mid-slab discontinuity

    Input:
        type alpha = float
        alpha = Angle between vertical and ray (in degrees)
        type x0 = float
        x0 = EW coordinate of the source (in m)
        type y0 = float
        y0 = NS coordinate of the source (in m)
        type d = float
        d = Depth of the source (in m)
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
        type t = float
        t = Corresponding travel time
    """
    # Checking input data
    assert (R1 == 'S' or R1 == 'P'), "Ray 1 must be a P or an S wave!"
    assert (R2 == 'S' or R2 == 'P'), "Ray 2 must be a P or an S wave!"
    assert (R3 == 'S' or R3 == 'P'), "Ray 3 must be a P or an S wave!"
    # Compute dipping in the source-receiver vertical plane
    dip = computeDip(x0, y0, x, y)
    # Compute incidence angles of rays
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
    beta = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * V2 / V1)
    gamma = (180.0 / pi) * asin(sin(beta * pi /180.0) * V3 / V2)
    # Downward travel time in upper oceanic crust
    D1 = hUOC * cos(dip * pi / 180.0)
    L1 = D1 / cos((alpha - dip) * pi /180.0)
    t1 = L1 / V1
    # Upward travel time in upper oceanic crust
    L2 = D1 / cos(beta * pi / 180.0)
    t2 = L2 / V2
    # Upward travel time in continental crust
    D2 = D1 * tan((alpha - dip) * pi / 180)
    D4 = D2 * sin(dip * pi / 180.0)
    D5 = D1 * tan(beta * pi / 180)
    D7 = D5 * sin(dip * pi / 180.0)
    D8 = d - D4 - D7
    L3 = D8 / cos((gamma - dip) * pi / 180.0)
    t3 = L3 / V3
    t = t1 + t2 + t3
    return t

def computeTravelTime5(alpha, x0, y0, d, x, y, R1='S', R2='S', R3='S', \
    R4='S', R5='S'):
    """
    Compute the travel time between the source and the receiver
    with one reflection at the Moho

    Input:
        type alpha = float
        alpha = Angle between vertical and ray (in degrees)
        type x0 = float
        x0 = EW coordinate of the source (in m)
        type y0 = float
        y0 = NS coordinate of the source (in m)
        type d = float
        d = Depth of the source (in m)
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
        type t = float
        t = Corresponding travel time
    """
    # Checking input data
    assert (R1 == 'S' or R1 == 'P'), "Ray 1 must be a P or an S wave!"
    assert (R2 == 'S' or R2 == 'P'), "Ray 2 must be a P or an S wave!"
    assert (R3 == 'S' or R3 == 'P'), "Ray 3 must be a P or an S wave!"
    assert (R4 == 'S' or R4 == 'P'), "Ray 4 must be a P or an S wave!"
    assert (R5 == 'S' or R5 == 'P'), "Ray 5 must be a P or an S wave!"
    # Compute dipping in the source-receiver vertical plane
    dip = computeDip(x0, y0, x, y)
    # Compute incidence angles of rays
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
    beta = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * V2 / V1)
    gamma = (180.0 / pi) * asin(sin(beta * pi /180.0) * V3 / V2)
    delta = (180.0 / pi) * asin(sin(gamma * pi / 180.0) * V4 / V3)
    epsilon = (180.0 / pi) * asin(sin(delta * pi / 180.0) * V5 / V4)
    # Downward travel time in upper oceanic crust
    D1 = hUOC * cos(dip * pi / 180.0)
    L1 = D1 / cos((alpha - dip) * pi / 180.0)
    t1 = L1 / V1
    # Downward travel time in lower oceanic crust
    D5 = hLOC * cos(dip * pi / 180.0)
    L2 = D5 / cos(beta * pi / 180.0)
    t2 = L2 / V2
    # Upward travel time in lower oceanic crust
    L3 = D5 / cos(gamma * pi / 180.0)
    t3 = L3 / V3
    # Upward travel time in upper oceanic crust
    L4 = D1 / cos(delta * pi / 180.0)
    t4 = L4 / V4
    # Upward travel time in continental crust
    D2 = D1 * tan((alpha - dip) * pi / 180)
    D4 = D2 * sin(dip * pi / 180.0)
    D6 = D5 * tan(beta * pi / 180)
    D8 = D6 * sin(dip * pi / 180.0)
    D9 = D5 * tan(gamma * pi / 180.0)
    D11 = D9 * sin(dip * pi / 180.0)
    D12 = D1 * tan(gamma * pi / 180.0)
    D14 = D12 * sin(dip * pi / 180.0)
    D15 = d - D4 - D8 - D11 - D14
    L5 = D15 / cos((epsilon - dip) * pi / 180.0)
    t5 = L5 / V5
    t = t1 + t2 + t3 + t4 + t5
    return t
