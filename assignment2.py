"""
FIT2004 2022 semester 2 assignment 2

"""

__authour__ = "Pin Hen Song"

from typing import List, Tuple
from math import inf
import heapq


#task 1
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
        self.num_loc = len(roads) #number of edge

        #create directed weightwd graph
        self.graph = [[] for _ in range(self.num_loc)] #create vertices, O(V)
        for loc in roads: #connect edges, O(E)
            u, v, w = loc
            self.graph[u].append((v,w))

        #reversed directed weighted graph
        self.graph_rev = [[] for _ in range(self.num_loc)] #create vertices, O(V)
        for loc in roads: #connect edges, O(E)
            u, v, w = loc
            self.graph_rev[v].append((u,w))
        
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
        
        start_to_cafe_pred, start_to_cafe_dist= self.shortest_path(start, False)#shortest path to cafe
        cafe_to_end_pred, cafe_to_end_dist = self.shortest_path(end, True) #shortest from cafe to destination
    
        
        #find shortest distance
        min_dist = inf
        for cafe in self.cafes:
            cafe_id, wait_time = cafe
            total_dist = start_to_cafe_dist[cafe_id] + cafe_to_end_dist[cafe_id] + wait_time
            if total_dist < min_dist:
                min_dist = total_dist
                cafe_chosen = cafe_id
        
        path_start_to_cafe = self.find_path(start_to_cafe_pred, cafe_chosen)
        path_cafe_to_end = self.find_path(cafe_to_end_pred, cafe_chosen)

        shortest_path = path_start_to_cafe[::-1][0:-1] + path_cafe_to_end

        return shortest_path

    def find_path(self, pred: List, end):
        path = [end]
        parent = pred[end]
        while parent != None:
            path.append(parent)
            parent = pred[parent]
            
        return path
        

    def shortest_path(self, start, rev: bool):
        #predecessor
        pred = [None for _ in range(self.num_loc)]
        dist = [inf for _ in range(self.num_loc)]

        #distance
        dist[start] = 0 #start = 0 
        
        #grpah
        graph = self.graph_rev if rev else self.graph #start to cafe or end to cafe

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
            for adj in graph[u]: #for neighbour of u
                v, w = adj
                #relax
                if dist[u] + w < dist[v]: #if new distance < current distance
                    dist[v] = dist[u] + w #update distance
                    pred[v] = u #update predecessor
                    heapq.heappush(p_queue,(dist[v], int(v))) #add v into queue

        return pred, dist

#task 2




if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    
    roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
            (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
            (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
    cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
    #mygraph = RoadGraph(roads, cafes)
    #print(mygraph.routing(1, 7), '\n') #[1, 7]
    #print(mygraph.routing(7, 8), '\n') #[7, 8]
    #print(mygraph.routing(1, 3), '\n') #[1, 5, 6, 3]
    #print(mygraph.routing(1, 4), '\n') #[1, 5, 6, 4]
    #print(mygraph.routing(3, 4), '\n') #[3, 4, 8, 7, 3, 4]
