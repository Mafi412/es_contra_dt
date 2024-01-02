import heapq
from collections import deque


class BehaviorArchive:
    def __init__(self, max_size, num_of_nearest_neighbors_to_average):
        self.k = num_of_nearest_neighbors_to_average
        self.archive = deque(maxlen=max_size)
        
    def get_novelty_score_for_behavior(self, behavior_characteristic):
        distances = (behavior_characteristic.compare_to(archive_member) for archive_member in self.archive)
        if self.k == 1:
            return min(distances)
        elif self.k >= len(self.archive):
            k_smallest_distances = list(distances)
        else:
            k_smallest_distances = heapq.nsmallest(self.k, distances)
            
        average_distance_to_k_nearest_neighbors = sum(k_smallest_distances) / len(k_smallest_distances)
        
        return average_distance_to_k_nearest_neighbors
    
    def add(self, behavior_characteristic):
        self.archive.append(behavior_characteristic)
        
    def extend(self, iterable_of_behavior_characteristics):
        self.archive.extend(iterable_of_behavior_characteristics)
