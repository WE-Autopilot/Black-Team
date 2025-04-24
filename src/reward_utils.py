import numpy as np

def get_progress(wp, point1, point2):
    n_waypoints = len(wp)
    
    #calculate all segment vectors and lengths at once using roll
    next_wp = np.roll(wp, -1, axis=0)  #gets each waypoint's next waypoint
    segments = next_wp - wp  #all segment vectors
    segment_lengths = np.linalg.norm(segments, axis=1)  #all segment lengths
    
    #precompute cumulative distances around the track
    cumulative_distances = np.zeros(n_waypoints + 1)
    cumulative_distances[1:] = np.cumsum(segment_lengths)
    total_track_length = cumulative_distances[-1]
    
    #find positions of both points on the track
    seg1_idx, pos1 = project_point_to_track(wp, segments, segment_lengths, cumulative_distances, point1)
    seg2_idx, pos2 = project_point_to_track(wp, segments, segment_lengths, cumulative_distances, point2)
    
    #calculate distances in both directions using modular arithmetic
    forward_dist = (pos2 - pos1) % total_track_length
    backward_dist = (pos1 - pos2) % total_track_length
    
    #return the shorter distance (negative if backward)
    if forward_dist <= backward_dist:
        return forward_dist
    else:
        return -backward_dist

def project_point_to_track(wp, segments, segment_lengths, cumulative_distances, point):
    #reshape point for broadcasting with arrays
    point_reshaped = point.reshape(1, 2)
    
    #calculate vectors from each waypoint to the point
    vectors_to_point = point_reshaped - wp
    
    #calculate normalized segment directions
    segment_directions = segments / segment_lengths[:, np.newaxis]
    
    #project point onto all segments at once using dot product
    projection_dots = np.sum(vectors_to_point * segment_directions, axis=1)
    
    #clamp to segment lengths
    clamped_dots = np.clip(projection_dots, 0, segment_lengths)
    
    #calculate all projected points
    projected_points = wp + clamped_dots[:, np.newaxis] * segment_directions
    
    #calculate distances to all projections
    distances = np.linalg.norm(point_reshaped - projected_points, axis=1)
    
    #find closest segment
    best_segment = np.argmin(distances)
    projection_distance = clamped_dots[best_segment]
    
    #calculate absolute position on track
    position = cumulative_distances[best_segment] + projection_distance
    
    return best_segment, position