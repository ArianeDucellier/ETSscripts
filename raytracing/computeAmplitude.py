#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy

from math import pi, cos, sin, asin

from data import VsUOC, VpUOC, rhoUOC, VsLOC, VpLOC, rhoLOC, \
                 VsCC, VpCC, rhoCC, VsOM, VpOM, rhoOM

from misc import computeDip, computeM, computeN

def computeAmplitude3SH(alpha, x0, y0, x, y):
    """Compute the amplitude of the wave at the receiver
    Input:
        alpha = angle between vertical and ray (in degrees)
        x0 = EW coordinate of the source (in m)
        y0 = NS coordinate of the source (in m)
        x = EW coordinate of the station (in m)
        y = NS coordinate of the station (in m)
    Output:
        A = corresponding amplitude
    """
    # Compute dipping in the source-receiver vertical plane
    dip = computeDip(x0, y0, x, y)
    # Amplitude of upward wave in upper oceanic crust
    Vs1 = VsUOC
    Vs2 = VsLOC
    rho1 = rhoUOC
    rho2 = rhoLOC
    j1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < Vs1 / Vs2:
        j2 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vs2 / Vs1)
    else:
        j2 = 90.0
    A1 = (rho1 * Vs1 * cos(j1 * pi / 180.0)
        - rho2 * Vs2 * cos(j2 * pi / 180.0)) / \
         (rho1 * Vs1 * cos(j1 * pi / 180.0) \
        + rho2 * Vs2 * cos(j2 * pi / 180.0))
    # Amplitude of upward wave in continental crust
    Vs1 = VsCC
    Vs2 = VsUOC
    rho1 = rhoCC
    rho2 = rhoUOC
    if abs(sin((alpha - dip) * pi / 180.0)) < Vs2 / Vs1:
        j1 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vs1 / Vs2)
    else:
        j1 = 90.0
    j2 = alpha - dip
    A2 = 2.0 * rho2 * Vs2 * cos(j2 * pi / 180.0) / \
              (rho1 * Vs1 * cos(j1 * pi / 180.0) \
             + rho2 * Vs2 * cos(j2 * pi / 180.0))
    A = A1 * A2
    return A

def computeAmplitude5SH(alpha, x0, y0, x, y):
    """Compute the amplitude of the wave at the receiver
    Input:
        alpha = angle between vertical and ray (in degrees)
        x0 = EW coordinate of the source (in m)
        y0 = NS coordinate of the source (in m)
        x = EW coordinate of the station (in m)
        y = NS coordinate of the station (in m)
    Output:
        A = corresponding amplitude
    """
    # Compute dipping in the source-receiver vertical plane
    dip = computeDip(x0, y0, x, y)
    # Compute incidence angles of rays
    beta = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * VsLOC / VsUOC)
    # Amplitude of downward wave in lower oceanic crust
    Vs1 = VsUOC
    Vs2 = VsLOC
    rho1 = rhoUOC
    rho2 = rhoLOC
    j1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < Vs1 / Vs2:
        j2 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vs2 / Vs1)
    else:
        j2 = 90.0
    A1 = 2.0 * rho1 * Vs1 * cos(j1 * pi / 180.0) / \
              (rho1 * Vs1 * cos(j1 * pi / 180.0) \
             + rho2 * Vs2 * cos(j2 * pi / 180.0))
    # Amplitude of upward wave in lower oceanic crust
    Vs1 = VsLOC
    Vs2 = VsOM
    rho1 = rhoLOC
    rho2 = rhoOM
    j1 = beta
    if abs(sin(beta * pi / 180)) < Vs1 / Vs2:
        j2 = (180.0 / pi) * asin(sin(beta * pi /180.0) * Vs2 / Vs1)
    else:
        j2 = 90.0
    A2 = (rho1 * Vs1 * cos(j1 * pi / 180.0)
        - rho2 * Vs2 * cos(j2 * pi / 180.0)) / \
         (rho1 * Vs1 * cos(j1 * pi / 180.0) \
        + rho2 * Vs2 * cos(j2 * pi / 180.0))
    # Amplitude of upward wave in upper oceanic crust
    Vs1 = VsUOC
    Vs2 = VsLOC
    rho1 = rhoUOC
    rho2 = rhoLOC
    j1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < Vs1 / Vs2:
        j2 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vs2 / Vs1)
    else:
        j2 = 90.0
    A3 = 2.0 * rho2 * Vs2 * cos(j2 * pi / 180.0) / \
              (rho1 * Vs1 * cos(j1 * pi / 180.0) \
             + rho2 * Vs2 * cos(j2 * pi / 180.0))
    # Amplitude of upward wave in continental crust
    Vs1 = VsCC
    Vs2 = VsUOC
    rho1 = rhoCC
    rho2 = rhoUOC
    if abs(sin((alpha - dip) * pi / 180.0)) < Vs2 / Vs1:
        j1 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vs1 / Vs2)
    else:
        j1 = 90.0
    j2 = alpha - dip
    A4 = 2.0 * rho2 * Vs2 * cos(j2 * pi / 180.0) / \
              (rho1 * Vs1 * cos(j1 * pi / 180.0) \
             + rho2 * Vs2 * cos(j2 * pi / 180.0))
    A = A1 * A2 * A3 * A4
    return A

def computeAmplitude3PSV(alpha, x0, y0, x, y, R1='S', R2='S', R3='S'):
    """Compute the amplitude of the wave at the receiver
    Input:
        alpha = angle between vertical and ray (in degrees)
        x0 = EW coordinate of the source (in m)
        y0 = NS coordinate of the source (in m)
        x = EW coordinate of the station (in m)
        y = NS coordinate of the station (in m)
        R1 = type of the downgoing ray in UOC (S or P)
        R2 = type of the upgoing ray in UOC (S or P)
        R3 = type of the upgoing ray in CC (S or P)
    Output:
        A = corresponding amplitude
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
    # Compute incidence angles of rays
    beta = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * V2 / V1)
    # Amplitude of upward wave in upper oceanic crust
    Vs1 = VsUOC
    Vs2 = VsLOC
    Vp1 = VpUOC
    Vp2 = VpLOC
    rho1 = rhoUOC
    rho2 = rhoLOC
    i1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < V1 / Vp2:
        i2 = (180.0 / pi) * asin(sin((alpha - dip) * pi / 180.0) * Vp2 / V1)
    else:
        i2 = 90.0
    j1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < V1 / Vs2:
        j2 = (180.0 / pi) * asin(sin((alpha - dip) * pi / 180.0) * Vs2 / V1)
    else:
        j2 = 90.0
    p = sin((alpha - dip) * pi / 180.0) / V1
    M = computeM(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    N = computeN(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    MN = numpy.dot(numpy.linalg.inv(M), N)
    if R1 == 'S':
        if R2 == 'S':
            A1 = MN[1, 1]
        else:
            A1 = MN[0, 1]
    else:
        if R2 == 'S':
            A1 = MN[1, 0]
        else:
            A1 = MN[0, 0]
    # Amplitude of upward wave in continental crust
    Vs1 = VsCC
    Vs2 = VsUOC
    Vp1 = VpCC
    Vp2 = VpUOC
    rho1 = rhoCC
    rho2 = rhoUOC
    i2 = beta
    if abs(sin(beta * pi / 180.0)) < V2 / Vp1:
        i1 = (180.0 / pi) * asin(sin(beta * pi / 180.0) * Vp1 / V2)
    else:
        i1 = 90.0
    j2 = beta
    if abs(sin(beta * pi / 180.0)) < V2 / Vs1:
        j1 = (180.0 / pi) * asin(sin(beta * pi / 180.0) * Vs1 / V2)
    else:
        j1 = 90.0
    p = sin(beta * pi / 180.0) / V2
    M = computeM(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    N = computeN(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    MN = numpy.dot(numpy.linalg.inv(M), N)
    if R2 == 'S':
        if R3 == 'S':
            A2 = MN[1, 3]
        else:
            A2 = MN[0, 3]
    else:
        if R3 == 'S':
            A2 = MN[1, 2]
        else:
            A2 = MN[0, 2]
    A = A1 * A2
    return A

def computeAmplitude5PSV(alpha, x0, y0, x, y, \
    R1='S', R2='S', R3='S', R4='S', R5='S'):
    """Compute the amplitude of the wave at the receiver
    Input:
        alpha = angle between vertical and ray (in degrees)
        x0 = EW coordinate of the source (in m)
        y0 = NS coordinate of the source (in m)
        x = EW coordinate of the station (in m)
        y = NS coordinate of the station (in m)
        R1 = type of the downgoing ray in UOC (S or P)
        R2 = type of the downgoing ray in LOC (S or P)
        R3 = type of the upgoing ray in LOC (S or P)
        R4 = type of the upgoing ray in UOC (S or P)
        R5 = type of the upgoing ray in CC (S or P)
    Output:
        A = corresponding amplitude
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
    beta = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * V2 / V1)
    gamma = (180.0 / pi) * asin(sin(beta * pi /180.0) * V3 / V2)
    delta = (180.0 / pi) * asin(sin(gamma * pi / 180.0) * V4 / V3)
    # Amplitude of downward wave in lower oceanic crust
    Vs1 = VsUOC
    Vs2 = VsLOC
    Vp1 = VpUOC
    Vp2 = VpLOC
    rho1 = rhoUOC
    rho2 = rhoLOC
    i1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < V1 / Vp2:
        i2 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vp2 / V1)
    else:
        i2 = 90.0
    j1 = alpha - dip
    if abs(sin((alpha - dip) * pi / 180)) < V1 / Vs2:
        j2 = (180.0 / pi) * asin(sin((alpha - dip) * pi /180.0) * Vs2 / V1)
    else:
        j2 = 90.0
    p = sin((alpha - dip) * pi / 180.0) / V1
    M = computeM(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    N = computeN(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    MN = numpy.dot(numpy.linalg.inv(M), N)
    if R1 == 'S':
        if R2 == 'S':
            A1 = MN[3, 1]
        else:
            A1 = MN[2, 1]
    else:
        if R2 == 'S':
            A1 = MN[3, 0]
        else:
            A1 = MN[2, 0]
    # Amplitude of upward wave in lower oceanic crust
    Vs1 = VsLOC
    Vs2 = VsOM
    Vp1 = VpLOC
    Vp2 = VpOM
    rho1 = rhoLOC
    rho2 = rhoOM
    i1 = beta
    if abs(sin(beta * pi / 180)) < V2 / Vp2:
        i2 = (180.0 / pi) * asin(sin(beta * pi /180.0) * Vp2 / V2)
    else:
        i2 = 90.0
    j1 = beta
    if abs(sin(beta * pi / 180)) < V2 / Vs2:
        j2 = (180.0 / pi) * asin(sin(beta * pi /180.0) * Vs2 / V2)
    else:
        j2 = 90.0
    p = sin(beta * pi / 180.0) / V2
    M = computeM(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    N = computeN(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    MN = numpy.dot(numpy.linalg.inv(M), N)
    if R2 == 'S':
        if R3 == 'S':
            A2 = MN[1, 1]
        else:
            A2 = MN[0, 1]
    else:
        if R3 == 'S':
            A2 = MN[1, 0]
        else:
            A2 = MN[0, 0]
    # Amplitude of upward wave in upper oceanic crust
    Vs1 = VsUOC
    Vs2 = VsLOC
    Vp1 = VpUOC
    Vp2 = VpLOC
    rho1 = rhoUOC
    rho2 = rhoLOC
    if abs(sin(gamma * pi / 180)) < V3 / Vp1:
        i1 = (180.0 / pi) * asin(sin(gamma * pi /180.0) * Vp1 / V3)
    else:
        i1 = 90.0
    i2 = gamma
    if abs(sin(gamma * pi / 180)) < V3 / Vs1:
        j1 = (180.0 / pi) * asin(sin(gamma * pi /180.0) * Vs1 / V3)
    else:
        j1 = 90.0
    j2 = gamma
    p = sin(gamma * pi / 180.0) / V3
    M = computeM(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    N = computeN(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    MN = numpy.dot(numpy.linalg.inv(M), N)
    if R3 == 'S':
        if R4 == 'S':
            A3 = MN[1, 3]
        else:
            A3 = MN[0, 3]
    else:
        if R4 == 'S':
            A3 = MN[1, 2]
        else:
            A3 = MN[0, 2]
    # Amplitude of upward wave in continental crust
    Vs1 = VsCC
    Vs2 = VsUOC
    Vp1 = VpCC
    Vp2 = VpUOC
    rho1 = rhoCC
    rho2 = rhoUOC
    if abs(sin(delta * pi / 180)) < V4 / Vp1:
        i1 = (180.0 / pi) * asin(sin(delta * pi /180.0) * Vp1 / V4)
    else:
        i1 = 90.0
    i2 = delta
    if abs(sin(delta * pi / 180)) < V4 / Vs1:
        j1 = (180.0 / pi) * asin(sin(delta * pi /180.0) * Vs1 / V4)
    else:
        j1 = 90.0
    j2 = delta
    p = sin(delta * pi / 180.0) / V4
    M = computeM(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    N = computeN(Vs1, Vp1, rho1, Vs2, Vp2, rho2, i1, i2, j1, j2, p)
    MN = numpy.dot(numpy.linalg.inv(M), N)
    if R4 == 'S':
        if R5 == 'S':
            A4 = MN[1, 3]
        else:
            A4 = MN[0, 3]
    else:
        if R5 == 'S':
            A4 = MN[1, 2]
        else:
            A4 = MN[0, 2]
    A = A1 * A2 * A3 * A4
    return A
