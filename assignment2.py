"""
FIT2004 2022 semester 2 assignment 2

"""

__authour__ = "Pin Hen Song"

from typing import List, Tuple
from math import inf
import heapq

class LocationVertex:
    def __init__(self)-> None:
        """
        location constructor
        :input:
            arg1: id, location id
        :time complexity: O(1)
        :aux space complexity:
        """
        self.adjacent = []
        self.wait_time = -1

class RoadGraph:
    def __init__(self, roads: List[Tuple], cafes:List[Tuple])->None:
        """
        graph constructor 
        :input:
            arg1: roads, list of tuples (u, v, w), where
                 u is the starting location ID for a road
                 v is the ending location ID for a road
                 w is the time taken to travel from location u to location v
            arg2: cafes, list of tuples (location, waiting_time), where
                 location is the location id of the cafe
                 waiting_time is the waiting time for a coffee in the cafe
        :output, return or postcondition:
        :time complexity: O(2V+2E) = 2O(V+E) = O(V+E)
        :aux space complexity:

        >>> roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
        ...         (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
        ...         (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        >>> cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        >>> g = RoadGraph(roads, cafes)
        """
        #find the number of locations, O(E)
        self.num_loc = 0
        for i in range(len(roads)):
            loc_id =  roads[i][0] or roads[i][1] #larger number of location id
            if loc_id+1 > self.num_loc:
                self.num_loc = loc_id+1 #update number of vertices

        #create vertices, O(V)
        self.graph = [LocationVertex() for _ in range(self.num_loc)]

        #connect edges, O(E)
        for loc in roads:
            u, v, w = loc
            self.graph[u].adjacent.append((v,w))

        #record cafe, worst case scenario: number of cafes = V, O(V)
        for cafe in cafes:
            cafe_id, wait_time = cafe
            self.graph[cafe_id].wait_time = wait_time
        
        self.cafes = cafes
        

    def routing(self, start: int, end: int):
        """
        location constructor
        :input:
            arg1: id, location id
        :time complexity: O(1)
        :aux space complexity:

        >>> roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
        ...         (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
        ...         (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        >>> cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        >>> mygraph = RoadGraph(roads, cafes)
        >>> mygraph.routing(1, 7)
        [1, 7]
        >>> mygraph.routing(7, 8)
        [7, 8]
        >>> mygraph.routing(1, 3)
        [1, 5, 6, 3]
        >>> mygraph.routing(1, 4)
        [1, 5, 6, 4]
        >>> mygraph.routing(3, 4)
        [3, 4, 8, 7, 3, 4]
        """
        
        start_to_cafe, cafe_to_end = [], []
        for cafe in self.cafes:
            start_to_cafe.append(self.shortest_path(start, cafe[0]))
            cafe_to_end.append(self.shortest_path(cafe[0], end))
        
        min_dist = inf
        for i in range(len(self.cafes)):
            if start_to_cafe[i][1] + cafe_to_end[i][1] + self.cafes[i][1] < min_dist:
                min_dist = start_to_cafe[i][1] + cafe_to_end[i][1] + self.cafes[i][1]
        
        for i in range(len(self.cafes)):
            if start_to_cafe[i][1] + cafe_to_end[i][1] + self.cafes[i][1] == min_dist:
                shortest_path = start_to_cafe[i][0][:-1] #***
                shortest_path += cafe_to_end[i][0]
                break
        
        return shortest_path

    def shortest_path(self, start, end):
        #predecessor
        pred = [None for _ in range(self.num_loc)]
        dist = [inf for _ in range(self.num_loc)]

        #distance
        dist[start] = 0 #start = 0 

        #priority queue
        p_queue = [(inf,i) for i in range(self.num_loc)]
        p_queue = [(0,start)] #start = 0 
        heapq.heapify(p_queue)
        p_queue_mark = [False for _ in range(self.num_loc)] #mark if a vertex has been searched
        

        while bool(p_queue):
            u = heapq.heappop(p_queue)[1] #pop
            while p_queue_mark[u] and bool(p_queue): #pop until found element that has never been popped
                u = heapq.heappop(p_queue)[1]
            p_queue_mark[u] = True #mark popped vertex

            #relax
            for adj in self.graph[u].adjacent: #for neighbour of u
                v, w = adj
                #relax
                if dist[u] + w < dist[v]: #if new distance < current distance
                    dist[v] = dist[u] + w #update distance
                    pred[v] = u #update predecessor
                    heapq.heappush(p_queue,(dist[v], int(v))) #add v into queue

        path = [end]
        u = end
        while pred[u] != None:
            path.append(pred[u])
            u = pred[u]
        return path[::-1], dist[end] #***complexity 

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    """
    roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
            (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
            (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
    cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
    mygraph = RoadGraph(roads, cafes)
    print(mygraph.routing(1, 7), '\n') #[1, 7]
    print(mygraph.routing(7, 8), '\n') #[7, 8]
    print(mygraph.routing(1, 3), '\n') #[1, 5, 6, 3]
    print(mygraph.routing(1, 4), '\n') #[1, 5, 6, 4]
    print(mygraph.routing(3, 4), '\n') #[3, 4, 8, 7, 3, 4]
    """