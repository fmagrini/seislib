#cython: language_level=3
"""
@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com


The module assumes a unit Earth
"""


cimport cython
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport pi as PI
from libc.math cimport sqrt



@cdivision
cdef double radians(double deg):
    return deg * PI / 180


@cdivision
cdef double degrees(double rad):
    return rad * 180 / PI


@boundscheck(False)
@wraparound(False)
cdef void cross_product_3d(double[3] v1, double[3] v2, double[3] result):
    result[0] = v1[1]*v2[2] - v1[2]*v2[1]
    result[1] = v1[2]*v2[0] - v1[0]*v2[2]
    result[2] = v1[0]*v2[1] - v1[1]*v2[0]


@boundscheck(False)
@wraparound(False)
cdef double norm_3d(double[3] v):
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

