"""
Miscellaneous functions to be used in the other modules
"""

import numpy

from math import pi, sqrt, pow, cos, sin, tan, atan, atan2

from data import phi, delta

def latLon2xy(lat, lon, lat0=0.0, lon0=0.0):
    """
    Compute the distance in m from an origin at (lat0, lon0)

    Input:
        type lat = 1D numpy array
        lat = Latitude of the points
        type lon = 1D numpy array
        lon = Longitude of the points
        type lat0 = float
        lat0 = Latitude of origin
        type lon0 = float
        lon0 = Longitude of origin
    Output:
        type x = 1D numpy array
        x = Distance in EW direction (in m)
        type y = 1D numpy array
        y = Distance in NS direction (in m)
    """
    # Earth's radius and ellipticity
    a = 6378.136
    e = 0.006694470
    # Corresponding grid step in EW and NS directions
    dx = (pi / 180.0) * a * numpy.cos(lat0 * pi / 180.0) / \
         numpy.sqrt(1.0 - e * e * numpy.sin(lat0 * pi / 180.0) * \
         numpy.sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / \
         ((1.0 - e * e * numpy.sin(lat0 * pi / 180.0) * \
         numpy.sin(lat0 * pi / 180.0)) ** 1.5)
    # Compute distances (in m)
    x = 1000.0 * (lon - lon0) * dx
    y = 1000.0 * (lat - lat0) * dy
    return (x, y)

def computeDip(x0, y0, x, y):
    """
    Compute the plate dipping in the station-source direction

    Input:
        type x0 = float
        x0 = EW coordinate of the source (in m)
        type y0 = float
        y0 = NS coordinate of the source (in m)
        type x = float
        x = EW coordinate of the station (in m)
        type y = float
        y = NS coordinate of the station (in m)        
    Output:
        type dip = float
        dip = Slope of the plate boundary (in degrees)
    """
    # Compute angle between station-source direction and dipping direction
    if sqrt((x - x0) ** 2.0 + (y - y0) ** 2.0) < 1.0:
        dip = delta
    else:
        # beta varies between -pi and pi
        beta = atan2(y0 - y, x0 - x)
        # alpha varies between 0 and 2pi
        alpha = pi / 2.0 - beta
        if alpha < 0.0:
            alpha = alpha + 2.0 * pi
        # gamma varies between -pi and pi
        gamma = phi * pi / 180.0 + beta
        if gamma <  - pi:
            gamma = gamma + 2.0 * pi
        if gamma >= pi:
            gamma = gamma - 2.0 * pi
        # Compute dip: If the station is up-dip of the source, dip is positive
        # If the source is up-dip of the station, dip is negative
        dip = atan(tan(delta * pi /180.0) * cos(gamma)) * 180.0 / pi
    return dip

def computeM(b1, a1, r1, b2, a2, r2, i1, i2, j1, j2, p):
    """Compute 4 x 4 matrix M from Aki and Richards (eq. 5.36 and Fig. 5.9) 
    Input:
        b1 = S-wave velocity in upper medium (in m/s)
        a1 = P-wave velocity in upper medium (in m/s)
        r1 = density in upper medium (in kg/m3)
        b2 = S-wave velocity in lower medium (in m/s)
        a2 = P-wave velocity in lower medium (in m/s)
        r2 = density in lower medium (in kg/m3)
        i1 = incidence angle of P-wave in upper medium (in degrees)
        i2 = incidence angle of P-wave in lower medium (in degrees)
        j1 = incidence angle of S-wave in upper medium (in degrees)
        j2 = incidence angle of S-wave in lower medium (in degrees)
        p = ray parameter (in s/m)
    Output:
        4 x 4 numpy array
    """
    i1 = i1 * pi / 180.0
    i2 = i2 * pi / 180.0
    j1 = j1 * pi / 180.0
    j2 = j2 * pi / 180.0
    M = numpy.zeros((4, 4))
    M[0, 0] = - a1 * p
    M[0, 1] = - cos(j1)
    M[0, 2] = a2 * p
    M[0, 3] = cos(j2)
    M[1, 0] = cos(i1)
    M[1, 1] = - b1 * p
    M[1, 2] = cos(i2)
    M[1, 3] = - b2 * p
    M[2, 0] = 2.0 * r1 * pow(b1, 2.0) * p * cos(i1)
    M[2, 1] = r1 * b1 * (1.0 - 2.0 * pow(b1, 2.0) * pow(p, 2.0))
    M[2, 2] = 2.0 * r2 * pow(b2, 2.0) * p * cos(i2)
    M[2, 3] = r2 * b2 * (1.0 - 2.0 * pow(b2, 2.0) * pow(p, 2.0))
    M[3, 0] = - r1 * a1 * (1.0 - 2.0 * pow(b1, 2.0) * pow(p, 2.0))
    M[3, 1] = 2.0 * r1 * pow(b1, 2.0) * p * cos(j1)
    M[3, 2] = r2 * a2 * (1.0 - 2.0 * pow(b2, 2.0) * pow(p, 2.0))
    M[3, 3] = - 2.0 * r2 * pow(b2, 2.0) * p * cos(j2)
    return M

def computeN(b1, a1, r1, b2, a2, r2, i1, i2, j1, j2, p):
    """Compute 4 x 4 matrix N from Aki and Richards (eq. 5.37 and Fig. 5.9) 
    Input:
        b1 = S-wave velocity in upper medium (in m/s)
        a1 = P-wave velocity in upper medium (in m/s)
        r1 = density in upper medium (in kg/m3)
        b2 = S-wave velocity in lower medium (in m/s)
        a2 = P-wave velocity in lower medium (in m/s)
        r2 = density in lower medium (in kg/m3)
        i1 = incidence angle of P-wave in upper medium (in degrees)
        i2 = incidence angle of P-wave in lower medium (in degrees)
        j1 = incidence angle of S-wave in upper medium (in degrees)
        j2 = incidence angle of S-wave in lower medium (in degrees)
        p = ray parameter
    Output:
        4 x 4 numpy array
    """
    i1 = i1 * pi / 180.0
    i2 = i2 * pi / 180.0
    j1 = j1 * pi / 180.0
    j2 = j2 * pi / 180.0
    N = numpy.zeros((4, 4))
    N[0, 0] = a1 * p
    N[0, 1] = cos(j1)
    N[0, 2] = - a2 * p
    N[0, 3] = - cos(j2)
    N[1, 0] = cos(i1)
    N[1, 1] = - b1 * p
    N[1, 2] = cos(i2)
    N[1, 3] = - b2 * p
    N[2, 0] = 2.0 * r1 * pow(b1, 2.0) * p * cos(i1)
    N[2, 1] = r1 * b1 * (1.0 - 2.0 * pow(b1, 2.0) * pow(p, 2.0))
    N[2, 2] = 2.0 * r2 * pow(b2, 2.0) * p * cos(i2)
    N[2, 3] = r2 * b2 * (1.0 - 2.0 * pow(b2, 2.0) * pow(p, 2.0))
    N[3, 0] = r1 * a1 * (1.0 - 2.0 * pow(b1, 2.0) * pow(p, 2.0))
    N[3, 1] = - 2.0 * r1 * pow(b1, 2.0) * p * cos(j1)
    N[3, 2] = - r2 * a2 * (1.0 - 2.0 * pow(b2, 2.0) * pow(p, 2.0))
    N[3, 3] = 2.0 * r2 * pow(b2, 2.0) * p * cos(j2)
    return N

def computeInitAmp(x0, y0, x, y, d, wavetype, alpha=0):
    """
    Compute the amplitudes of the P-, SV- and SH-waves at the tremor source

    Input:
        typex0 = float
        x0 = EW coordinate of the source (in m)
        type y0 = float
        y0 = NS coordinate of the source (in m)
        type x = float
        x = EW coordinate of the station (in m)
        type y = float
        y = NS coordinate of the station (in m)    
        type d = float
        d = Depth of the tremor source (in m)
        type wavetype = string
        wavetype = Direct (D) or reflected (R) wave
        type alpha = float
        alpha = Angle between the vertical and the direction of propagation
    Output:
        type amp = 1D numpy array of length 3
        amp = Values of the amplitude for the P-, SV-, and SH-waves
    """
    # Checking input data
    assert (wavetype == 'D' or wavetype == 'R'), \
        "Wave must be direct (D) or reflected (R)!"
    # Computing the moment tensor in (X,Y,Z) coordinates
    u = numpy.array([- cos(delta * pi / 180.0) * cos(phi * pi / 180.0), \
                     cos(delta * pi / 180.0) * sin(phi * pi / 180.0), \
                     sin(delta * pi / 180.0)])
    nu = numpy.array([sin(delta * pi / 180.0) * cos(phi * pi / 180.0), \
                     - sin(delta * pi / 180.0) * sin(phi * pi / 180.0), \
                     cos(phi * pi / 180.0)])
    M_XYZ = numpy.array([[u[0] * nu[0] + u[0] * nu[0], \
                          u[0] * nu[1] + u[1] * nu[0], \
                          u[0] * nu[2] + u[2] * nu[0]], \
                         [u[1] * nu[0] + u[0] * nu[1], \
                          u[1] * nu[1] + u[1] * nu[1], \
                          u[1] * nu[2] + u[2] * nu[1]], \
                         [u[2] * nu[0] + u[0] * nu[2], \
                          u[2] * nu[1] + u[1] * nu[2], \
                          u[2] * nu[2] + u[2] * nu[2]]])
    # Converting to (R,T,Z) coordinates
    beta = atan2(y - y0, x - x0) * 180.0 / pi
    N = numpy.array([[cos(beta * pi / 180.0), sin(beta * pi / 180.0), 0], \
                     [- sin(beta * pi / 180.0), cos(beta * pi / 180.0), 0],
                     [0.0, 0.0, 1.0]])
    M_RTZ = numpy.dot(numpy.dot(N, M_XYZ), N.transpose())
    # Converting to (P,SV,SH) coordinates
    if wavetype == 'D':
        alpha = atan2(sqrt(pow(x - x0, 2.0) + pow(y - y0, 2.0)), d) \
            * 180.0 / pi
        N = numpy.array([[sin(alpha * pi /180.0), 0.0, \
                          cos(alpha * pi / 180.0)], \
                         [cos(alpha * pi / 180.0), 0.0, \
                          - sin(alpha * pi / 180.0)], \
                         [0.0, 1.0, 0.0]])
        M_PVH = numpy.dot(numpy.dot(N, M_RTZ), N.transpose())
    else:
        N = numpy.array([[sin(alpha * pi /180.0), 0, \
                          - cos(alpha * pi / 180.0)], \
                         [cos(alpha * pi / 180.0), 0, \
                          sin(alpha * pi / 180.0)], \
                         [0.0, - 1.0, 0.0]])
        M_PVH = numpy.dot(numpy.dot(N, M_RTZ), N.transpose())
    amp = numpy.array([M_PVH[0, 0], M_PVH[1, 0], M_PVH[2, 0]])
    return amp
