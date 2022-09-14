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
        :time complexity: O(2|V|+2|E|) = 2O(|V|+|E|) = O(|V|+|E|)
        :aux space complexity: adjacency list, O(|V|+|E|)

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
        find a shortest path from start to end that passes exactly one cafe
        :input:
            arg1: start, location id of starting point
            arg2: end, location id of destination
        :time complexity: O(|E|log|V|)
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

        if min_dist == inf: #no cafe or no reachable cafe 
            return None

        path_start_to_cafe = self.find_path(start_to_cafe_pred, start, cafe_chosen)
        path_cafe_to_end = self.find_path(cafe_to_end_pred, end, cafe_chosen)

        shortest_path = path_start_to_cafe[::-1][0:-1] + path_cafe_to_end
        return shortest_path

    def find_path(self, pred: List, start, end):
        """
        return path from start to end if there is one, else return None

        :input: 
            arg1: pred, predecessors list that keep track of parent node
                  in a shortest path
            arg2: start, location id of starting point
            arg3: end, location id of destination
        :time complexity: O(|E|)
        :aux space complexity: O(|V| + |E|), where len(pred) == |V|, len(path) == |E|
        """
        path = [end]
        parent = pred[end]
        while parent != None:
            path.append(parent)
            parent = pred[parent]

        if path[-1] != start:
            return None #can't be reached
        else: 
            return path
        

    def shortest_path(self, start, rev: bool):
        """
        Dijkstra algorithm that return predecessors list and return list

        :input: 
            arg1: start, location id of starting point
            arg2: rev, search on reversed direction graph if True,
                  else search on original graph
        :time complexity: O(|E|log|V|)
        :aux space complexity:
        """
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
def optimalRoute(downhillScores, start, finish):
    """
    location constructor
    :input:
        arg1: downhillScores
        arg2: start
        arg3: finish
    :time complexity: O(|D|) or O(|D|*|P|) 
    :aux space complexity:
    >>> downhillScores = [(0, 6, -500), (1, 4, 100), (1, 2, 300),
    ...                  (6, 3, -100), (6, 1, 200), (3, 4, 400), (3, 1, 400),
    ...                  (5, 6, 700), (5, 1, 1000), (4, 2, 100)]
    >>> start, finish = 6, 2
    >>> optimalRoute(downhillScores, start, finish)
    [6, 3, 1, 2]
    """
    g = graph(downhillScores)
    topo_order = topo_sort_dfs(g) #topological order list
    dist, pred = longest_path(g, topo_order) #find optimal route, i.e. longest path
    return find_path(pred,start,finish)

def find_path(pred, start, finish):
    """
    location constructor
    :input:
        arg1: pred
        arg2: start
        arg3: finish
    :time complexity: O(|D|)
    :aux space complexity:
    """
    path = [finish]
    parent = pred[finish]
    while bool(parent) and parent != start:
        path.append(parent)
        parent = pred[parent]
    
    if not bool(parent):
        return None
    else:
        path.append(start)
        path.reverse()
        return path
        
def longest_path(g, topo_order):
    """
    
    :input:
        arg1: g
        arg2: topo_order
    :time complexity: O(|D|+|P|) = O(|D|)
    :aux space complexity:
    """
    dist = [-inf for _ in range(len(g))] #list that hold the longest path
    dist[0] = 0

    pred = [None for _ in range(len(g))]
    for u in topo_order:
        for item in g[u]:
            v, w = item
            if w + dist[u] > dist[v]:
                dist[v] = w + dist[u]
                pred[v] = u
    return dist, pred


def topo_sort_dfs(g):
    """
    topological sort
    :input:
        arg1: g, graph
    :time complexity: O(|D| + |P|) = O(|D|) 
    :aux space complexity:
    """
    vst = [False for _ in range(len(g))]
    order = []
    for u in range(len(g)):
        if not vst[u]:
            topo_sort_dfs_aux(g, u, vst, order) 
    order.reverse()
    return order

def topo_sort_dfs_aux(g, u, vst, order):
    """
    topological sort aux
    :input:
        arg1: g, graph
        arg2: u
        arg3: vst, visited list
        arg4: order, topological order list
    :time complexity: O(|D| + |P|) = O(|D|) 
    :aux space complexity: dominanted by graph, O(|D|)
    """
    vst[u] = True
    for item in g[u]:
        v = item[0]
        if not vst[v]:
            topo_sort_dfs_aux(g, v, vst, order)
    order.append(u)

def graph(list):
    """
    graph constructor
    :input:
        arg1: list of tuple (u,v,w), 
              where u & v are vertex, w is weight
    :time complexity: O(|D| + |P|) = O(|D| + |D|) = O(|D|) 
    :aux space complexity: adjacency matrix, O(|D| + |P|) = O(|D|) 
    """
    
    n = 0 #number of vertex
    for item in list: #O(|D|)
        if item[0] > n: 
            n = item[0]
        if item[1] > n:
            n = item[1]

    g = [[] for _ in range(n+1)] #graph, O(|D|)
    for item in list:
        u,v,w = item
        g[u].append((v,w))
    return g

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    
    #roads = [(1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
    #        (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
    #        (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
    #cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
    #mygraph = RoadGraph(roads, cafes)
    #print(mygraph.routing(0, 1), '\n') #[1, 7]
    #print(mygraph.routing(7, 8), '\n') #[7, 8]
    #print(mygraph.routing(1, 3), '\n') #[1, 5, 6, 3]
    #print(mygraph.routing(1, 4), '\n') #[1, 5, 6, 4]
    #print(mygraph.routing(3, 4), '\n') #[3, 4, 8, 7, 3, 4]
    downhillScores = [(0, 6, -500), (1, 4, 100), (1, 2, 300),
                      (6, 3, -100), (6, 1, 200), (3, 4, 400), (3, 1, 400),
                      (5, 6, 700), (5, 1, 1000), (4, 2, 100)]
    print(optimalRoute(downhillScores, 6, 2))