#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from math import pi, cos, sin, tan, asin

from data import VsUOC, VpUOC, VsLOC, VpLOC, VsCC, VpCC, hUOC, hLOC

from misc import computeDip

def computeDistance3(alpha, x0, y0, d, x, y, R1='S', R2='S', R3='S'):
    """Compute the distance between source and receiver (in m)
    Input:
        alpha = angle between vertical and ray (in degrees)
        x0 = EW coordinate of the source (in m)
        y0 = NS coordinate of the source (in m)
        d = depth of the source (in m)
        x = EW coordinate of the station (in m)
        y = NS coordinate of the station (in m)
        R1 = type of the downgoing ray in UOC (S or P)
        R2 = type of the upgoing ray in UOC (S or P)
        R3 = type of the upgoing ray in CC (S or P)
    Output:
        X = corresponding distance
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
    # Get the distances
    D1 = hUOC * cos(dip * pi / 180.0)
    D2 = D1 * tan((alpha - dip) * pi / 180)
    D3 = D2 * cos(dip * pi / 180.0)
    D4 = D2 * sin(dip * pi / 180.0)
    D5 = D1 * tan(beta * pi / 180)
    D6 = D5 * cos(dip * pi / 180.0)
    D7 = D5 * sin(dip * pi / 180.0)
    D8 = d - D4 - D7
    D9 = D8 * tan((gamma - dip) * pi / 180.0)
    X = D3 + D6 + D9
    return X

def computeDistance5(alpha, x0, y0, d, x, y, \
    R1='S', R2='S', R3='S', R4='S', R5='S'):
    """Compute the distance between source and receiver
    Input:
        alpha = angle between vertical and ray (in degrees)
        x0 = EW coordinate of the source (in m)
        y0 = NS coordinate of the source (in m)
        d = depth of the source (in m)
        x = EW coordinate of the station (in m)
        y = NS coordinate of the station (in m)
        R1 = type of the downgoing ray in UOC (S or P)
        R2 = type of the downgoing ray in LOC (S or P)
        R3 = type of the upgoing ray in LOC (S or P)
        R4 = type of the upgoing ray in UOC (S or P)
        R5 = type of the upgoing ray in CC (S or P)
    Output:
        X = corresponding distance
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
    # Get the distances
    D1 = hUOC * cos(dip * pi / 180.0)
    D2 = D1 * tan((alpha - dip) * pi / 180)
    D3 = D2 * cos(dip * pi / 180.0)
    D4 = D2 * sin(dip * pi / 180.0)
    D5 = hLOC * cos(dip * pi / 180.0)
    D6 = D5 * tan(beta * pi / 180)
    D7 = D6 * cos(dip * pi / 180.0)
    D8 = D6 * sin(dip * pi / 180.0)
    D9 = D5 * tan(gamma * pi / 180.0)
    D10 = D9 * cos(dip * pi / 180.0)
    D11 = D9 * sin(dip * pi / 180.0)
    D12 = D1 * tan(gamma * pi / 180.0)
    D13 = D12 * cos(dip * pi / 180.0)
    D14 = D12 * sin(dip * pi / 180.0)
    D15 = d - D4 - D8 - D11 - D14
    D16 = tan((epsilon - dip) * pi / 180.0) * D15
    X = D3 + D7 + D10 + D13 + D16
    return X
 