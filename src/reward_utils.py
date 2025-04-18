import numpy as np

def get_progress(wp, point1, point2):
    n_waypoints = len(wp)
    
    #find the nearest segments and distances to waypoints
    seg1_idx, dist1_to_start = find_nearest_segment_and_distance(wp, point1)
    seg2_idx, dist2_to_start = find_nearest_segment_and_distance(wp, point2)
    
    #calculate segment lengths
    seg1_length = np.linalg.norm(wp[(seg1_idx + 1) % n_waypoints] - wp[seg1_idx])
    seg2_length = np.linalg.norm(wp[(seg2_idx + 1) % n_waypoints] - wp[seg2_idx])
    
    #calculate distances to end waypoints
    dist1_to_end = seg1_length - dist1_to_start
    dist2_to_end = seg2_length - dist2_to_start
    
    #calculate forward distance
    forward_dist = calculate_distance(wp, seg1_idx, dist1_to_start, dist1_to_end, 
                                      seg2_idx, dist2_to_start, dist2_to_end)
    
    #calculate backward distance
    backward_dist = calculate_distance(wp, seg2_idx, dist2_to_start, dist2_to_end,
                                       seg1_idx, dist1_to_start, dist1_to_end)
    
    #choose the shorter distance (negative if backward)
    if forward_dist <= backward_dist:
        return forward_dist
    else:
        return -backward_dist

def find_nearest_segment_and_distance(wp, point):
    n_waypoints = len(wp)
    min_dist = float('inf')
    best_segment = 0
    best_dist_to_start = 0
    
    for i in range(n_waypoints):
        start_wp = wp[i]
        end_wp = wp[(i + 1) % n_waypoints]
        
        #vector from segment start to end
        segment_vec = end_wp - start_wp
        segment_length = np.linalg.norm(segment_vec)
        segment_direction = segment_vec / segment_length
        
        #vector from segment start to point
        point_vec = point - start_wp
        
        #calculate projection parameter (t)
        t = np.dot(point_vec, segment_direction)
        t = max(0, min(segment_length, t))  # Clamp to segment
        
        #calculate projected point
        proj_point = start_wp + (t/segment_length) * segment_vec
        
        #distance from point to projection
        dist_to_proj = np.linalg.norm(point - proj_point)
        
        if dist_to_proj < min_dist:
            min_dist = dist_to_proj
            best_segment = i
            best_dist_to_start = t  #distance along segment to start
    
    return best_segment, best_dist_to_start

def calculate_distance(wp, seg1_idx, dist1_to_start, dist1_to_end, seg2_idx, dist2_to_start, dist2_to_end):
    n_waypoints = len(wp)
    total_dist = 0
    
    #if both points are on the same segment
    if seg1_idx == seg2_idx:
        return dist2_to_start - dist1_to_start
    
    #distance from point1 to end of its segment
    total_dist += dist1_to_end
    
    #distance through waypoints between segments
    current_idx = (seg1_idx + 1) % n_waypoints
    while current_idx != seg2_idx:
        next_idx = (current_idx + 1) % n_waypoints
        total_dist += np.linalg.norm(wp[next_idx] - wp[current_idx])
        current_idx = next_idx
    
    #distance from start of seg2 to point2
    total_dist += dist2_to_start
    
    return total_dist