#cython: language_level=3
"""
@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com


The module assumes a unit Earth
"""


cimport cython
from cython cimport cdivision, boundscheck, wraparound, binding
from libc.math cimport cos, sin, tan, acos, asin, atan, atan2
from libc.math cimport pi as PI
from libc.math cimport sqrt, fabs
from libcpp cimport bool as bool_cpp
from libcpp.list cimport list as cpplist
import numpy as np
cimport numpy as np
from seislib.tomography._ray_theory._math cimport radians, degrees
from seislib.tomography._ray_theory._math cimport cross_product_3d, norm_3d
cdef double TWOPI = 2 * PI


@boundscheck(False)
@wraparound(False)
cdef void cartesian_coordinates(double[2] latlon, double[3] result):
    """
    [lat, lon] in radians
    """
    result[0] = cos(latlon[0]) * cos(latlon[1])
    result[1] = cos(latlon[0]) * sin(latlon[1])
    result[2] = sin(latlon[0])


@boundscheck(False)
@wraparound(False)
cdef void spherical_coordinates(double[3] xyz, double[2] result):
    """
    cartesian coordinates: [x, y, z]
    """
    result[0] = asin(xyz[2])
    result[1] = atan2(xyz[1], xyz[0])


@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef void gc_arc_midpoint(double[2] latlon_1, double[2] latlon_2, double[2] midpoint):
    """
    v1, v2 : cartesian coordinates of the two points on the great circle
    """
    cdef double norm
    cdef double[3] xyz_1, xyz_2, xyz_midpoint
    cartesian_coordinates(latlon_1, xyz_1)
    cartesian_coordinates(latlon_2, xyz_2)
    xyz_midpoint[:] = [ xyz_1[0]+xyz_2[0], xyz_1[1]+xyz_2[1], xyz_1[2]+xyz_2[2] ]
    norm = norm_3d(xyz_midpoint)
    xyz_midpoint[:] = [ xyz_midpoint[0]/norm, xyz_midpoint[1]/norm, xyz_midpoint[2]/norm ]
    spherical_coordinates(xyz_midpoint, midpoint)
    

@boundscheck(False)
@wraparound(False)
cdef void great_circle_plane(double[2] r1, double[2] r2, double[3] params):
    """
    Plane through r1, r2 (lat, lon in radians), and origin sphere
    """
    params[0] = cos(r1[0])*sin(r2[0])*sin(r1[1]) - cos(r2[0])*sin(r1[0])*sin(r2[1])
    params[1] = cos(r2[0])*sin(r1[0])*cos(r2[1]) - cos(r1[0])*sin(r2[0])*cos(r1[1])
    params[2] = cos(r1[0]) * cos(r2[0]) * sin(r2[1]-r1[1])


@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef double great_circle_distance(double[2] latlon_1, double[2] latlon_2):
    """
    Calculate the great circle distance between two points 
    on the earth.
    [lat1, lon1], [lat2, lon2] : radians
    """
    cdef double lat1 = latlon_1[0]
    cdef double lon1 = latlon_1[1]
    cdef double lat2 = latlon_2[0]
    cdef double lon2 = latlon_2[1]
    cdef double dlon = lon1 - lon2
    cdef double dlat = lat1 - lat2
    return 2*asin(sqrt((sin((dlat)/2))**2 + cos(lat1)*cos(lat2)*(sin((dlon)/2))**2))
    

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef double gc_meridian_intersection(double[3] gc_plane, double parallel1, 
                                     double parallel2, double meridian):
    cdef double a_12 = gc_plane[0]
    cdef double b_12 = gc_plane[1]
    cdef double c_12 = gc_plane[2]               
    lat_intersect = -atan( a_12*cos(meridian)/c_12 + b_12*sin(meridian)/c_12 )
    if parallel1 <= lat_intersect <= parallel2: 
        return lat_intersect 
    return 9999


@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cdef void gc_parallel_intersection(double[2] r1, double[2] r2, double meridian1, 
                                   double meridian2, double parallel, double[2] lon_intersections):
    lon_intersections[:] = [9999, 9999]
    cdef double lat1 = r1[0]
    cdef double lon1 = r1[1]
    cdef double lat2 = r2[0]
    cdef double lon2 = r2[1]
    cdef double l12 = lon1 - lon2
    cdef double a, b, c, ab_norm
    cdef double lon, lon_intersection1, lon_intersection2
    a = sin(lat1)*cos(lat2)*cos(parallel)*sin(l12)
    b = sin(lat1)*cos(lat2)*cos(parallel)*cos(l12) - cos(lat1)*sin(lat2)*cos(parallel)
    c = cos(lat1)*cos(lat2)*sin(parallel)*sin(l12)
    lon = atan2(b, a)
    ab_norm = sqrt(a**2 + b**2)
    if (fabs(c) > ab_norm):
        return
    dlon = acos(c / ab_norm)
    lon_intersection1 = mod(lon1+dlon+lon+PI, TWOPI) - PI
    lon_intersection2 = mod(lon1-dlon+lon+PI, TWOPI) - PI
    if meridian1 <= lon_intersection1 <= meridian2:
        lon_intersections[0] = lon_intersection1
    if meridian1 <= lon_intersection2 <= meridian2:
        lon_intersections[1] = lon_intersection2


cpdef double mod(double a, double b):
    return a % b



