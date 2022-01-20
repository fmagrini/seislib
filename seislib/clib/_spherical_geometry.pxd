

cdef void cartesian_coordinates(double[2], double[3])

cdef void spherical_coordinates(double[3], double[2])

cdef void gc_arc_midpoint(double[2], double[2], double[2])

cdef void great_circle_plane(double[2], double[2], double[3])

cdef double great_circle_distance(double[2], double[2])

cdef double gc_meridian_intersection(double[3], double, double, double)
    
cdef void gc_parallel_intersection(double[2], double[2], double, double, double, double[2])

