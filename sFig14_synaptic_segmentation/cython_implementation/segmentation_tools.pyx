from collections import deque
import numpy
from random import randint
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def kernel_apply_3d( input_array, kernel_array ):
    assert len( input_array.shape ) == len( kernel_array.shape ) == 3
    
    output_array = numpy.zeros(input_array.shape, dtype=int )
    cdef int[:,:,:] output_data = output_array
    cdef int[:,:,:] kernel_data = kernel_array
    cdef int[:,:,:] input_data  = input_array
    
    cdef int xlim = input_array.shape[0]
    cdef int ylim = input_array.shape[1]
    cdef int zlim = input_array.shape[2]
    
    cdef int kernelx = kernel_array.shape[0]
    cdef int kernely = kernel_array.shape[1]
    cdef int kernelz = kernel_array.shape[2]
    
    cdef double accum = 0
    cdef int dz = 0
    cdef int dx = 0
    cdef int dy = 0
    
    cdef int x
    cdef int y
    cdef int z
    
    cdef int xi
    cdef int yi
    cdef int zi
    
    for x in range( xlim ):
        for y in range( ylim ):
            for z in range( zlim ):
                accum = 0
                for xi in range( kernelx ):
                    for yi in range( kernely ):
                        for zi in range( kernelz ):
                            dz, dx, dy = zi - (kernelz//2), xi - (kernelx//2), yi - (kernely//2)
                            dz, dx, dy = z + dz, x + dx, y + dy
                            
                            if dz < 0: dz *= -1
                            if dx < 0: dx *= -1
                            if dy < 0: dy *= -1
                                
                            if dz >= zlim: dz = 2 * zlim - dz - 1
                            if dx >= xlim: dx = 2 * xlim - dx - 1
                            if dy >= ylim: dy = 2 * ylim - dy - 1
                            
                            accum += kernel_data[xi, yi, zi]*input_data[dx,dy,dz]
                            
                output_data[x,y,z] = int( accum )
    return output_array

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def threshold_image( input_array, int limit ):
    cdef int zlim = input_array.shape[0]
    cdef int xlim = input_array.shape[1]
    cdef int ylim = input_array.shape[2]
    
    cdef int[:,:,:] array_data = input_array
    
    for z in range( zlim ):
        for x in range( xlim ):
            for y in range( ylim ):
                if array_data[z,x,y] <= limit:
                    array_data[z,x,y] = 0
                else:
                    array_data[z,x,y] = -1
                    
@cython.boundscheck(False)
@cython.wraparound(False)
def flood_threshold( input_array, counts_list, int upper_bound, int lower_bound ): #bounds inclusive
    cdef int[:,:,:] array_data = input_array
    cdef int vol, total
    
    cdef int zlim = input_array.shape[0]
    cdef int xlim = input_array.shape[1]
    cdef int ylim = input_array.shape[2]
    
    q = deque()
    for coord, voli in counts_list.items():
        vol = voli
        total = 0
        if vol > upper_bound or vol < lower_bound:
            # get rid of it
            q.append( coord )

            while q:
                pt = q.pop()
                if array_data[pt[0], pt[1], pt[2]] == 0:
                    continue

                array_data[pt[0], pt[1], pt[2]] = 0

                if pt[0]-1 >= 0:
                    q.append( (pt[0]-1, pt[1], pt[2]) )
                if pt[0]+1 != zlim:
                    q.append( (pt[0]+1, pt[1], pt[2]) )
                if pt[1]-1 >= 0:
                    q.append( (pt[0], pt[1]-1, pt[2]) )
                if pt[1]+1 != xlim:
                    q.append( (pt[0], pt[1]+1, pt[2]) )
                if pt[2]-1 >= 0:
                    q.append( (pt[0], pt[1], pt[2]-1) )
                if pt[2]+1 != ylim:
                    q.append( (pt[0], pt[1], pt[2]+1) )
                    
                total += 1 
                    
    out = {}
    for coor, voli in counts_list.items():
        vol = voli
        if vol > upper_bound or vol < lower_bound:
            continue
        out[ coor ] = voli
            
    return out
                 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def flood_fill( input_array ):
    '''
    This code takes an array of ints where 0 represents background and -1 represents target and uniquely
    labels each blob with a int code
    '''
    
    cdef int zlim = input_array.shape[0]
    cdef int xlim = input_array.shape[1]
    cdef int ylim = input_array.shape[2]
    cdef int[:,:,:] array_data = input_array
    cdef int color = randint(2,2147483647)
    cdef int total = 0
    
    q = deque()
    counts = dict()
    
    for z in range( zlim ):
        for x in range( xlim ):
            for y in range( ylim ):
                if array_data[z,x,y] == -1:
                    #color += 1
                    color = randint(2,2147483645)
                    total = 0
                    q.append( (z,x,y) )

                    while q:
                        pt = q.pop()
                        if array_data[pt[0], pt[1], pt[2]] != -1:
                            continue
                            
                        total += 1
                        array_data[pt[0], pt[1], pt[2]] = color

                        if pt[0]-1 >= 0:
                            q.append( (pt[0]-1, pt[1], pt[2]) )
                        if pt[0]+1 != zlim:
                            q.append( (pt[0]+1, pt[1], pt[2]) )
                        if pt[1]-1 >= 0:
                            q.append( (pt[0], pt[1]-1, pt[2]) )
                        if pt[1]+1 != xlim:
                            q.append( (pt[0], pt[1]+1, pt[2]) )
                        if pt[2]-1 >= 0:
                            q.append( (pt[0], pt[1], pt[2]-1) )
                        if pt[2]+1 != ylim:
                            q.append( (pt[0], pt[1], pt[2]+1) )
                    
                    counts[ (z,x,y) ] = total
    return counts

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def find_centroids( input_array, counts_list ): #bounds inclusive
    cdef int[:,:,:] array_data = input_array
    
    cdef int zlim = input_array.shape[0]
    cdef int xlim = input_array.shape[1]
    cdef int ylim = input_array.shape[2]
    
    cdef int xavg
    cdef int yavg
    cdef int zavg
    
    cdef int[3] pt
    
    out = {}
    q = deque()
    for coord, voli in list( counts_list.items() ):
        pts = set()
        
        q.append( coord )

        while q:
            pti = q.pop()
            pt = pti
            
            if pti in pts:
                continue
            if array_data[pt[0], pt[1], pt[2]] == 0:
                continue

            pts.add( pti )

            if pt[0]-1 >= 0:
                q.append( (pt[0]-1, pt[1], pt[2]) )
            if pt[0]+1 != zlim:
                q.append( (pt[0]+1, pt[1], pt[2]) )
            if pt[1]-1 >= 0:
                q.append( (pt[0], pt[1]-1, pt[2]) )
            if pt[1]+1 != xlim:
                q.append( (pt[0], pt[1]+1, pt[2]) )
            if pt[2]-1 >= 0:
                q.append( (pt[0], pt[1], pt[2]-1) )
            if pt[2]+1 != ylim:
                q.append( (pt[0], pt[1], pt[2]+1) )

        xavg = 0
        yavg = 0
        zavg = 0
            
        for pti in pts:
            xavg += pti[1]
            yavg += pti[2]
            zavg += pti[0]
            
        centroid = ( float(zavg)/len(pts), float(xavg)/len(pts), float(yavg)/len(pts) ) #zxy
        out[ centroid ] = voli
        
        del pts

    return out

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def find_closest( source_array, target_array, float z_scale ):
    out = dict()
    
    cdef float min_dist
    cdef float dist
    cdef float[3] min_coord
    
    cdef float iz, ix, iy
    cdef float jz, jx, jy
    
    for iz,ix,iy in source_array:
        min_dist = 100000000
        for jz,jx,jy in target_array:
                dist = (jx-ix)**2
                dist += (jy-iy)**2 
                dist += (z_scale*(jz-iz))**2
                dist = dist**0.5
                
                if dist < min_dist:
                    min_dist = dist
                    min_coord[0] = jz
                    min_coord[1] = jx
                    min_coord[2] = jy
                    #min_coord = (jz,jx,jy)
                    
        out[ (iz,ix,iy) ] = (min_dist, tuple(x for x in min_coord))
    return out

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def look_around( input_array, input_array2, counts_list ): #bounds inclusive
    cdef int[:,:,:] array_data = input_array
    cdef int[:,:,:] array_data2 = input_array2
    
    cdef int zlim = input_array.shape[0]
    cdef int xlim = input_array.shape[1]
    cdef int ylim = input_array.shape[2]
    
    cdef int xavg 
    cdef int yavg
    cdef int zavg
    
    cdef int[3] pt
    
    cdef float total_intensity
    cdef int total_count
    
    out = {}
    q = deque()
    for coord, voli in list( counts_list.items() ):
        pts = set()
        q.append( coord )
        total_intensity = 0
        total_count = 0

        while q:
            pti = q.pop()
            pt = pti
            
            if pti in pts:
                continue
            if array_data[pt[0], pt[1], pt[2]] == 0:
                total_intensity += array_data2[ pt[0], pt[1], pt[2] ]
                total_count += 1
                continue

            pts.add( pti )

            if pt[0]-1 >= 0:
                q.append( (pt[0]-1, pt[1], pt[2]) )
            if pt[0]+1 != zlim:
                q.append( (pt[0]+1, pt[1], pt[2]) )
            if pt[1]-1 >= 0:
                q.append( (pt[0], pt[1]-1, pt[2]) )
            if pt[1]+1 != xlim:
                q.append( (pt[0], pt[1]+1, pt[2]) )
            if pt[2]-1 >= 0:
                q.append( (pt[0], pt[1], pt[2]-1) )
            if pt[2]+1 != ylim:
                q.append( (pt[0], pt[1], pt[2]+1) )
                
        xavg = 0
        yavg = 0
        zavg = 0
            
        for pti in pts:
            xavg += pti[1]
            yavg += pti[2]
            zavg += pti[0]
            
        centroid = ( float(zavg)/len(pts), float(xavg)/len(pts), float(yavg)/len(pts) ) #zxy
        out[ centroid ] = total_intensity / total_count
                
        del pts

    return out