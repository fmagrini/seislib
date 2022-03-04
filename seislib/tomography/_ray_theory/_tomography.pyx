#cython: language_level=3
"""
@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com
"""

cimport cython
from cython cimport cdivision, boundscheck, wraparound, nonecheck
from cython.parallel import prange, parallel
from libc.math cimport cos, sin, tan, acos, asin, atan, atan2
from libc.math cimport pi as PI
from libc.math cimport NAN
from libc.math cimport sqrt, fabs, fmin, fmax
from libcpp cimport bool as bool_cpp
from libcpp.list cimport list as cpplist
from libcpp.vector cimport vector as cppvector
from seislib.tomography._ray_theory._spherical_geometry cimport cartesian_coordinates, spherical_coordinates
from seislib.tomography._ray_theory._spherical_geometry cimport great_circle_distance, gc_arc_midpoint, great_circle_plane
from seislib.tomography._ray_theory._spherical_geometry cimport gc_meridian_intersection, gc_parallel_intersection
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
cdef double TWOPI = 2 * PI
cdef extern from "math.h":
    float INFINITY
   

@boundscheck(False)
@wraparound(False) 
cdef void gc_pixel_intersections(double[2] r1, double[2] r2, double[3] gc_plane, 
                                 double[4][2] pixel_intersections,  double parallel1, 
                                 double meridian1, double parallel2, double meridian2,
                                 double[2] lon_intersections, double[2] midpoint, 
                                 double midpoint_distance, double[2] tmp_2d):
    """
    Input:
        r1, r2 : coordinates (lat, lon) of the two stations in radians
        gc_plane: parameters of the plane passing through the great circle and the origin of the sphere
        parallel1(2), meridian1(2) : boundaries of the pixel, in radians
    Returns:
        pixel_intersections : intersections between gc_plane and pixel. (2, 2) array --> ([[lat1, lon1], [lat2, lon2]])
        frac_gc : fraction of the great circle arc connecting r1 and r2 passing through the pixel
    """
#    cdef double[2][2] pixel_intersections
    cdef ssize_t intersection_idx, ilon, size_lons=2
    cdef double lat_intersect, lon_intersect
    pixel_intersections[0][:] = [9999, 9999]
    pixel_intersections[1][:] = [9999, 9999]
    pixel_intersections[2][:] = [9999, 9999]
    pixel_intersections[3][:] = [9999, 9999]
    intersection_idx = 0
    #check the presence of intersections with meridians
    lat_intersect = gc_meridian_intersection(gc_plane, parallel1, parallel2, meridian1)
    if not lat_intersect == 9999:
        tmp_2d[:] = [lat_intersect, meridian1]
        if great_circle_distance(tmp_2d, midpoint) - midpoint_distance <= 1e-10:
            pixel_intersections[intersection_idx][0] = lat_intersect
            pixel_intersections[intersection_idx][1] = meridian1
            intersection_idx += 1
    lat_intersect = gc_meridian_intersection(gc_plane, parallel1, parallel2, meridian2)
    if not lat_intersect == 9999:
        tmp_2d[:] = [lat_intersect, meridian2]
        if great_circle_distance(tmp_2d, midpoint) - midpoint_distance <= 1e-10:
            pixel_intersections[intersection_idx][0] = lat_intersect
            pixel_intersections[intersection_idx][1] = meridian2
            intersection_idx += 1
    #check the presence of intersections with parallels
    gc_parallel_intersection(r1, r2, meridian1, meridian2, parallel1, lon_intersections)
    for ilon in range(size_lons):
        if not lon_intersections[ilon] == 9999:
            tmp_2d[:] = [parallel1, lon_intersections[ilon]]
            if great_circle_distance(tmp_2d, midpoint) - midpoint_distance <= 1e-10:
                pixel_intersections[intersection_idx][0] = parallel1
                pixel_intersections[intersection_idx][1] = lon_intersections[ilon]
                intersection_idx += 1
    gc_parallel_intersection(r1, r2, meridian1, meridian2, parallel2, lon_intersections)
    for ilon in range(size_lons):
        if not lon_intersections[ilon] == 9999:
            tmp_2d[:] = [parallel2, lon_intersections[ilon]]
            if great_circle_distance(tmp_2d, midpoint) - midpoint_distance <= 1e-10:
                pixel_intersections[intersection_idx][0] = parallel2
                pixel_intersections[intersection_idx][1] = lon_intersections[ilon]
                intersection_idx += 1


@boundscheck(False)
@wraparound(False) 
cpdef ssize_t _pixel_index(double lat, double lon, double[:, ::1] mesh,
                           double mesh_latmax, double mesh_lonmax):
    cdef ssize_t ipixel
    cdef ssize_t rows = mesh.shape[0]
    if lat == mesh_latmax:
        lat -= 0.000001
    if lon == mesh_lonmax:
        lon -= 0.000001
    for ipixel in range(rows):
        if mesh[ipixel, 0] <= lat < mesh[ipixel, 1]: #parallel1<=lat<parallel2
            if mesh[ipixel, 2] <= lon < mesh[ipixel, 3]: #meridian1<=lon<meridian2
                return ipixel
     

@boundscheck(False)
@wraparound(False) 
cdef void bound_parallels(double[2] r1, double[2] r2, double[2] latmin_latmax):
    cdef double lat1 = r1[0]
    cdef double lon1 = r1[1]
    cdef double lat2 = r2[0]
    cdef double lon2 = r2[1]
    cdef double latmin12 = fmin(lat1, lat2)
    cdef double latmax12 = fmax(lat1, lat2)
    #Worst cases, to account for the curvature of the gc
    cdef double[2] coords_1
    cdef double[2] coords_2
    ######
    cdef double[2] midpoint_max, midpoint_min
    coords_1[:] = [latmin12, lon1]
    coords_2[:] = [latmin12, lon2]
    gc_arc_midpoint(coords_1, coords_2, midpoint_min)
    latmin_latmax[0] = fmin(latmin12, midpoint_min[0])
    coords_1[:] = [latmax12, lon1]
    coords_2[:] = [latmax12, lon2]
    gc_arc_midpoint(coords_1, coords_2, midpoint_max)
    latmin_latmax[1] = fmax(latmax12, midpoint_max[0])
    

@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cpdef _compile_coefficients(double[:, ::1] data_coords, double[:, ::1] mesh,
                            double mesh_latmax, double mesh_lonmax, 
                            bool_cpp refine=False, double[:, ::1] coeff_matrix=None):
    """
    Input:
        Data coordinates : (m x 4) Matrix, whose cols are (lat1, lon1, lat2, lon2)
        mesh : (n x 4) Matrix, describing a mesh of n pixels identified by
               (parallel1, parallel2, meridian1, meridian2)
    Output:
        Matrix of the coefficients : (m x n). For the ith data point (row), the numerical
        value of its jth col represents the fraction of the great circle connecting
        (lat1, lon1) and (lat2, lon2) passing through the pixel
    """

    cdef ssize_t data_size = data_coords.shape[0]
    cdef ssize_t mesh_size = mesh.shape[0]
    cdef ssize_t idata, ipixel, idx
    cdef ssize_t ipixel_start, ipixel_end #pixels of the two points
    cdef double gc_distance, midpoint_distance, intersection_distance
    cdef double parallel1, parallel2, meridian1, meridian2
    cdef double latmin, latmax #to restrict the research for intersections according to each data point
    cdef double lonmin, lonmax
    cdef double coeff
    cdef bool_cpp dlon_lessthan_pi
    cdef double[2] latmin_latmax, latlon_1, latlon_2
    cdef double[2] midpoint #between (latlon_1, latlon_2), in spherical coordinates
    cdef double[2] lon_intersections, tmp_2d
    cdef double[3] gc_plane
    cdef double[4] distances
    cdef int[4] sorted_idx
    cdef double[4][2] intersections #gc intersections with pixel
    if coeff_matrix is None:
        coeff_matrix = np.zeros((data_size, mesh_size), dtype=np.double)
    for idata in range(data_size):  
        latlon_1[:] = [data_coords[idata, 0], data_coords[idata, 1]]
        latlon_2[:] = [data_coords[idata, 2], data_coords[idata, 3]]
        # if the two points lie withing the same pixel, the coefficient is 1
        ipixel_start = _pixel_index(latlon_1[0], latlon_1[1], mesh, mesh_latmax, mesh_lonmax)
        ipixel_end = _pixel_index(latlon_2[0], latlon_2[1], mesh, mesh_latmax, mesh_lonmax)
        #calculate all parameters used later on in the process of finding the great 
        #circle intersections with the pixels of the mesh
        gc_arc_midpoint(latlon_1, latlon_2, midpoint)
        gc_distance = great_circle_distance(latlon_1, latlon_2)
        if gc_distance == 0:
            continue
        midpoint_distance = gc_distance / 2
        great_circle_plane(latlon_1, latlon_2, gc_plane)
        #bound the mesh to (latmin, latmax), (lonmin, lonmax) so as to skip many iterations
        bound_parallels(latlon_1, latlon_2, latmin_latmax)
        latmin = latmin_latmax[0]
        latmax = latmin_latmax[1]
        lonmin = fmin(latlon_1[1], latlon_2[1])
        lonmax = fmax(latlon_1[1], latlon_2[1])
        dlon_lessthan_pi = lonmax-lonmin < PI
        for ipixel in range(mesh_size):
            if refine:
                if coeff_matrix[idata, ipixel] != 9999:
                    continue
                coeff_matrix[idata, ipixel] = 0
            parallel1 = mesh[ipixel, 0]
            parallel2 = mesh[ipixel, 1]
            meridian1 = mesh[ipixel, 2]
            meridian2 = mesh[ipixel, 3]
            ####if these conditions are not verified, the pixel is not interested by the gc path####
            if parallel1>latmax or parallel2<latmin:
                continue
            if dlon_lessthan_pi:
                if meridian1>lonmax or meridian2<lonmin:
                    continue
            else:
                #if lonmax-lonmin > pi, the great circle will pass through the opposide "side" of the earth
                if meridian1>lonmin and meridian2<lonmax:
                    continue
            gc_pixel_intersections(r1=latlon_1, 
                                   r2=latlon_2, 
                                   gc_plane=gc_plane, 
                                   pixel_intersections=intersections,
                                   lon_intersections=lon_intersections,
                                   parallel1=parallel1, 
                                   parallel2=parallel2, 
                                   meridian1=meridian1, 
                                   meridian2=meridian2,
                                   midpoint=midpoint, 
                                   midpoint_distance=midpoint_distance,
                                   tmp_2d=tmp_2d)

            coeff = gc_fraction(latlon_1=latlon_1, 
                                latlon_2=latlon_2, 
                                intersections=intersections, 
                                ipixel=ipixel, 
                                ipixel_start=ipixel_start, 
                                ipixel_end=ipixel_end, 
                                gc_distance=gc_distance,
                                distances=distances, 
                                sorted_idx=sorted_idx)
            if coeff > 0: # Does not access memory unnecessarily
                coeff_matrix[idata, ipixel] = coeff
    return np.asarray(coeff_matrix)
                

@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cdef double gc_fraction(double[2] latlon_1, double[2] latlon_2, double[4][2] intersections, 
                        ssize_t ipixel, ssize_t ipixel_start, ssize_t ipixel_end, 
                        double gc_distance, double[4] distances, int[4] sorted_idx):
    cdef int no_intersections = 0
    cdef ssize_t i, size=4
    cdef double length = 0
    distances[:] = [9999, 9999, 9999, 9999]
    
    for i in range(size):
        if intersections[i][0] == 9999:
            break
        no_intersections += 1
        
    if no_intersections == 0:
        if ipixel_start==ipixel_end and ipixel==ipixel_start:
            return 1
        return 0
    
    elif no_intersections == 1:
        if ipixel == ipixel_start:
            return great_circle_distance(latlon_1, intersections[0]) / gc_distance
        if ipixel == ipixel_end:
            return great_circle_distance(latlon_2, intersections[0]) / gc_distance
        return 0
    
    elif no_intersections == 2:
        return great_circle_distance(intersections[0], intersections[1]) / gc_distance
    
    elif no_intersections == 3:
        
        if ipixel == ipixel_start:
            for i in range(no_intersections):
                distances[i] = great_circle_distance(latlon_1, intersections[i])    
            argsort(distances, sorted_idx, no_intersections)
            
            if ipixel == ipixel_end:
                length = distances[1]
                length += great_circle_distance(intersections[sorted_idx[2]], 
                                                latlon_2)
            else:
                length = distances[0]
                length += great_circle_distance(intersections[sorted_idx[1]], 
                                                intersections[sorted_idx[2]])
        
        elif ipixel == ipixel_end:
            for i in range(no_intersections):
                distances[i] = great_circle_distance(latlon_2, intersections[i])    
            argsort(distances, sorted_idx, no_intersections)    
            length = distances[0]
            length += great_circle_distance(intersections[sorted_idx[1]], 
                                            intersections[sorted_idx[2]])
        
        return length / gc_distance
    
    elif no_intersections == 4:
        for i in range(no_intersections):
            distances[i] = great_circle_distance(latlon_1, intersections[i])
        argsort(distances, sorted_idx, no_intersections)
        length = great_circle_distance(intersections[sorted_idx[0]], 
                                       intersections[sorted_idx[1]])
        length += great_circle_distance(intersections[sorted_idx[2]], 
                                        intersections[sorted_idx[3]])
        return length / gc_distance
        

@boundscheck(False)
@wraparound(False) 
cpdef _raypaths_per_pixel(double[:, ::1] coeff_matrix):
    cdef ssize_t rows = coeff_matrix.shape[0] #number of data
    cdef ssize_t cols = coeff_matrix.shape[1] #number of pixels
    cdef ssize_t row, col
    cdef np.int32_t[::1] counts = np.zeros(cols, dtype=np.int32)
    for row in range(rows):
        for col in range(cols):
            if coeff_matrix[row, col] > 0:
                counts[col] += 1
    return counts


@boundscheck(False)
@wraparound(False) 
cdef bool_cpp is_inside_region(double lat, double lon, double[:] region):
    if region[0]<=lat<=region[1] and region[2]<=lon<=region[3]:
        return True
    return False


@cdivision(True)
@boundscheck(False)
@wraparound(False) 
def _refine_parameterization(double[:, ::1] mesh, double[:, ::1] coeff_matrix, 
                             int hitcounts=100, double[:] region_to_refine=None):
    cdef np.int32_t[::1] rays_count = _raypaths_per_pixel(coeff_matrix)
    cdef ssize_t data_size = coeff_matrix.shape[0]
    cdef ssize_t mesh_size = mesh.shape[0] #number of pixels
    cdef ssize_t newmesh_size = mesh_size
    cdef ssize_t i_pixel, i_data
    cdef ssize_t i_newpixel = 0
    cdef bool_cpp focus_on_a_region = False
    cdef bool_cpp inside_region = True
    cdef double parallel1, parallel2, meridian1, meridian2
    cdef double coeff
    cdef double lat_pixel, lon_pixel
    
    if region_to_refine is not None:
        focus_on_a_region = True
    for i_pixel in range(mesh_size):
        if focus_on_a_region:
            lat_pixel = (mesh[i_pixel, 0] + mesh[i_pixel, 1]) / 2
            lon_pixel = (mesh[i_pixel, 2] + mesh[i_pixel, 3]) / 2
            inside_region = is_inside_region(lat=lat_pixel,
                                             lon=lon_pixel,
                                             region=region_to_refine)
        if rays_count[i_pixel]>=hitcounts and inside_region:
            # Remove one pixel and add 4 -- the pixels itself divided by four
            newmesh_size += 3
            
    cdef double[:, ::1] newmesh = np.zeros((newmesh_size, 4), dtype=np.double)
    cdef double[:, ::1] newcoeff_matrix = np.zeros((data_size, newmesh_size), dtype=np.double)
   
    if region_to_refine is not None:
        focus_on_a_region = True
    
    for i_pixel in range(mesh_size):
        if focus_on_a_region:
            lat_pixel = (mesh[i_pixel, 0] + mesh[i_pixel, 1]) / 2
            lon_pixel = (mesh[i_pixel, 2] + mesh[i_pixel, 3]) / 2
            inside_region = is_inside_region(lat=lat_pixel,
                                             lon=lon_pixel,
                                             region=region_to_refine)
                
        if rays_count[i_pixel]<hitcounts or not inside_region:
            newmesh[i_newpixel, 0] = mesh[i_pixel, 0]
            newmesh[i_newpixel, 1] = mesh[i_pixel, 1]
            newmesh[i_newpixel, 2] = mesh[i_pixel, 2]
            newmesh[i_newpixel, 3] = mesh[i_pixel, 3]
            for i_data in prange(data_size, nogil=True):
                coeff = coeff_matrix[i_data, i_pixel]
                if coeff > 0: # Does not access memory unnecessarily
                    newcoeff_matrix[i_data, i_newpixel] += coeff
            i_newpixel += 1
        else:
            parallel1 = mesh[i_pixel, 0]
            parallel2 = mesh[i_pixel, 1]
            meridian1 = mesh[i_pixel, 2]
            meridian2 = mesh[i_pixel, 3]    
            split_pixel(newmesh, i_newpixel, parallel1, parallel2, meridian1, meridian2)
            for i_data in prange(data_size, nogil=True):
                newcoeff_matrix[i_data, i_newpixel] = 9999
                newcoeff_matrix[i_data, i_newpixel+1] = 9999
                newcoeff_matrix[i_data, i_newpixel+2] = 9999
                newcoeff_matrix[i_data, i_newpixel+3] = 9999
            i_newpixel += 4
    return np.asarray(newmesh), newcoeff_matrix
    

@boundscheck(False)
@wraparound(False) 
def _select_parameters(double[:, ::1] coeff_matrix, np.int32_t[::1] indexes):
    cdef ssize_t data_size = coeff_matrix.shape[0]
    cdef ssize_t newmesh_size = indexes.size
    cdef ssize_t newmesh_index, mesh_index, i_data
    cdef double[:, ::1] newcoeff_matrix = np.zeros((data_size, newmesh_size), dtype=np.double)
    
    for newmesh_index in range(newmesh_size):
        mesh_index = indexes[newmesh_index]
        for i_data in prange(data_size, nogil=True):
            if coeff_matrix[i_data, mesh_index] > 0:
                newcoeff_matrix[i_data, newmesh_index] = coeff_matrix[i_data, mesh_index]                
    return np.asarray(newcoeff_matrix)


@boundscheck(False)
@wraparound(False) 
cdef void split_pixel(double[:, ::1] newmesh, ssize_t i_newpixel, double parallel1,
                      double parallel2, double meridian1, double meridian2):
    newmesh[i_newpixel, 0] = (parallel1 + parallel2) / 2
    newmesh[i_newpixel, 1] = parallel2
    newmesh[i_newpixel, 2] = meridian1
    newmesh[i_newpixel, 3] = (meridian1 + meridian2) / 2
    newmesh[i_newpixel+1, 0] = (parallel1 + parallel2) / 2
    newmesh[i_newpixel+1, 1] = parallel2
    newmesh[i_newpixel+1, 2] = (meridian1 + meridian2) / 2
    newmesh[i_newpixel+1, 3] = meridian2
    newmesh[i_newpixel+2, 0] = parallel1
    newmesh[i_newpixel+2, 1] = (parallel1 + parallel2) / 2
    newmesh[i_newpixel+2, 2] = meridian1
    newmesh[i_newpixel+2, 3] = (meridian1 + meridian2) / 2
    newmesh[i_newpixel+3, 0] = parallel1
    newmesh[i_newpixel+3, 1] = (parallel1 + parallel2) / 2
    newmesh[i_newpixel+3, 2] = (meridian1 + meridian2) / 2
    newmesh[i_newpixel+3, 3] = meridian2


@boundscheck(False)
@wraparound(False) 
cdef void argsort(double[4] arr, int[4] result, ssize_t size=4):
    """
    arr is modified
    """
    cdef bool_cpp done
    cdef ssize_t i
    cdef double tmp_a
    cdef int tmp_r
    for i in range(size):
        result[i] = i
    size -=1
    while 1:
        done = True
        for i in range(size):
            if arr[i+1] < arr[i]:
                tmp_a = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = tmp_a
                tmp_r = result[i]
                result[i] = result[i+1]
                result[i+1] = tmp_r
                done = False
        if done:
            return


@boundscheck(False)
@wraparound(False) 
cdef double get_fundamental_size(double[:, ::1] mesh) nogil:
    cdef ssize_t mesh_size = mesh.shape[0]
    cdef ssize_t ipixel
    cdef double fundamental_size = mesh[0, 1] - mesh[0, 0]
    cdef double size
    for ipixel in range(mesh_size):
        size = mesh[ipixel, 1] - mesh[ipixel, 0]
        if size < fundamental_size:
            fundamental_size = size
    return fundamental_size


@cdivision(True)
@boundscheck(False)
@wraparound(False) 
def _derivatives_lat_lon(double[:, ::1] mesh):
    cdef ssize_t mesh_size = mesh.shape[0]
    cdef ssize_t ipixel, jpixel
    cdef int size_lats_ajacent, size_lons_ajacent
    cdef int i
    cdef int nonzero_counter_lon, nonzero_counter_lat
    cdef double[:, ::1] G_lat = np.identity(mesh_size, dtype=np.double)
    cdef double[:, ::1] G_lon = np.identity(mesh_size, dtype=np.double)
    cdef double lat1_i, lat2_i, lon1_i, lon2_i
    cdef double lat1_j, lat2_j, lon1_j, lon2_j
    cdef double lat_size_i, lon_size_i, lat_size_j, lon_size_j
    cdef double fundamental_size = get_fundamental_size(mesh)
    cdef double fundamental_size_lon, n_fundamental_blocks, fundamental_ratio
    cdef double contact_size
    
    for ipixel in range(mesh_size):        
        lat1_i = mesh[ipixel, 0]
        lat2_i = mesh[ipixel, 1]
        lon1_i = mesh[ipixel, 2]
        lon2_i = mesh[ipixel, 3]
        lat_size_i = lat2_i - lat1_i
        lon_size_i = lon2_i - lon1_i
        fundamental_ratio = lat_size_i / fundamental_size
        n_fundamental_blocks = fundamental_ratio**2
        fundamental_size_lon = lon_size_i / fundamental_ratio
        G_lat[ipixel, ipixel] /= fundamental_ratio
        G_lon[ipixel, ipixel] /= fundamental_ratio
        for jpixel in prange(mesh_size, nogil=True):
            if ipixel == jpixel:
                continue
            lat1_j = mesh[jpixel, 0]
            lat2_j = mesh[jpixel, 1]
            lon1_j = mesh[jpixel, 2]
            lon2_j = mesh[jpixel, 3]
            lat_size_j = lat2_j - lat1_j
            lon_size_j = lon2_j - lon1_j
            
            if lat1_i == lat2_j:
                if lon1_j==lon1_i and lon2_j==lon2_i:
                    contact_size = lon_size_i
                elif lon1_j<=lon1_i<lon2_j:
                    if lon2_j < lon2_i:
                        contact_size = lon2_j - lon1_i
                    else:
                        contact_size = lon_size_i
                elif lon1_i<lon1_j and lon1_j<lon2_i<=lon2_j:
                    if lon2_j <= lon2_i:
                        contact_size = lon_size_j
                    else:
                        contact_size = lon2_i - lon1_j
                elif lon1_i<lon1_j and lon2_i>lon2_j:
                    contact_size = lon_size_j
                else:
                    contact_size = 0
                if contact_size > 0:
                    G_lat[ipixel, jpixel] = -contact_size / fundamental_size_lon / n_fundamental_blocks

            elif lon2_i == lon1_j:                
                if lat1_j==lat1_i and lat2_j==lat2_i:
                    contact_size = lat_size_i
                elif lat1_i<lat1_j<lat2_i and lat2_i==lat2_j \
                or lat1_i<lat2_j<lat2_i and lat1_i==lat1_j:
                    contact_size = lat_size_j
                elif lat1_j<lat1_i<lat2_j and lat2_i==lat2_j \
                or lat1_j<lat2_i<lat2_j and lat1_i==lat1_j:
                    contact_size = lat_size_i
                elif lat1_j<lat1_i and lat2_j>lat2_i:
                    contact_size = lat_size_i
                else:
                    contact_size = 0
                if contact_size > 0:
                    G_lon[ipixel, jpixel] = -contact_size / fundamental_size / n_fundamental_blocks
        
        # The derivative of the pixels at the E and S boundaries is set to zero
        nonzero_counter_lon = 0
        nonzero_counter_lat = 0
        for jpixel in range(mesh_size):
            if G_lon[ipixel, jpixel] != 0:
                nonzero_counter_lon += 1
            if G_lat[ipixel, jpixel] != 0:
                nonzero_counter_lat += 1
        if nonzero_counter_lon == 1:
            G_lon[ipixel, ipixel] = 0
        if nonzero_counter_lat == 1:
            G_lat[ipixel, ipixel] = 0        

    return np.asarray(G_lat)/fundamental_size, np.asarray(G_lon)/fundamental_size



