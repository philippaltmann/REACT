import math


def min_point_distance(p1, trajectory):
    min_dist = math.dist(p1, trajectory[0])
    for p_i in range(1, len(trajectory)):
        dist = math.dist(p1, trajectory[p_i])
        if dist < min_dist:
            min_dist = dist
    return min_dist


def one_way_distance(trajectory_1, trajectory_2):
    sum_of_min_point_distances = 0
    for p in trajectory_1:
        sum_of_min_point_distances += min_point_distance(p, trajectory_2)
    return (1/len(trajectory_1)) * sum_of_min_point_distances


# distance for grids
def two_way_distance(trajectory_1, trajectory_2):
    return 0.5 * (one_way_distance(trajectory_1, trajectory_2) + one_way_distance(trajectory_2, trajectory_1))
