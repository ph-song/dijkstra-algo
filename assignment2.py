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
        :postcondition: graph represented in adjacenct list is constructed 
        :time complexity: O(2|V|+2|E|) = 2O(|V|+|E|) = O(|V|+|E|),
                          where |V| is number of vetices, |E is number of edge
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
        

    def routing(self, start: int, end: int) -> List or None:
        """
        find a shortest path from start to end that passes exactly one cafe
        :input:
            arg1: start, location id of starting point
            arg2: end, location id of destination
        :time complexity: O(|E|log|V|), dominated by self.shortest_path()
                          where |V| is number of vetices, |E is number of edge
        :return: path from start to end that passes exactly on cafe,
                 which is represented in list
        :aux space complexity: O(|V|+|E|)

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
        if len(self.graph) == 0 or len(self.cafes) == 0:
            return None
        
        start_to_cafe_pred, start_to_cafe_dist= self.shortest_path(start, False)#shortest path from start to cafe #O(|E|log|V|)
        cafe_to_end_pred, cafe_to_end_dist = self.shortest_path(end, True) #shortest from cafe to destination #O(|E|log|V|)
    
        
        #find shortest distance
        min_dist = inf
        for cafe in self.cafes:
            cafe_id, wait_time = cafe
            total_dist = start_to_cafe_dist[cafe_id] + cafe_to_end_dist[cafe_id] + wait_time
            if total_dist < min_dist:
                min_dist = total_dist #update shortest path
                cafe_chosen = cafe_id #update cafe to buy coffee

        if min_dist == inf: #no cafe or no reachable cafe 
            return None

        #get path
        path_start_to_cafe = self.find_path(start_to_cafe_pred, start, cafe_chosen) #path from start to cafe
        path_cafe_to_end = self.find_path(cafe_to_end_pred, end, cafe_chosen) #path from end to cafe

        shortest_path = path_start_to_cafe[::-1] + path_cafe_to_end[1:] #worst case: all vertices are included in path O(|V|)
        return shortest_path

    def find_path(self, pred: List, start: int, end: int)-> List or None:
        """
        return path from start to end if there is one, else return None

        :input: 
            arg1: pred, predecessors list that keep track of parent of nodes
                  in a shortest path
            arg2: start, location id of starting point
            arg3: end, location id of destination
        :return: a list, which is a path from start to end if there is one, else return None
        :time complexity: O(|V|), worse case path include all vertex
        :aux space complexity: O(|V|), |V| is number of vertex
        """
        path = [end]
        parent = pred[end]
        while parent != None:
            path.append(parent)
            parent = pred[parent]

        if path[-1] != start: #can't be reached
            return None 
        else: 
            return path
        

    def shortest_path(self, start, rev: bool) ->Tuple[List]:
        """
        Dijkstra algorithm that return predecessors list and distance list

        :input: 
            arg1: start, location id of starting point
            arg2: rev, flag to indicate which graph to search
                  search reversed direction graph if True,
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
        heapq.heapify(p_queue) #O(|V|log|V|)
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
def optimalRoute(downhillScores, start, finish) -> List:
    """
    location constructor
    :input:
        arg1: downhillScores
        arg2: start
        arg3: finish
    :time complexity: O(|D|) where |D| is number of downhill segment, i.e. len(downhillScores)
    :aux space complexity:
    >>> downhillScores = [(0, 6, -500), (1, 4, 100), (1, 2, 300),
    ...                  (6, 3, -100), (6, 1, 200), (3, 4, 400), (3, 1, 400),
    ...                  (5, 6, 700), (5, 1, 1000), (4, 2, 100)]
    >>> start, finish = 6, 2
    >>> optimalRoute(downhillScores, start, finish)
    [6, 3, 1, 2]
    """
    if start == finish:
        return [start]

    g = graph(downhillScores)
    topo_order = topo_sort_dfs(g) #O(|D|), topological order list
    dist, pred = longest_path(g, topo_order) #O(|D|), find optimal route, i.e. longest path
    return find_path(pred,start,finish) #O(|D|)

def find_path(pred: List, start: int, finish: int)-> List or None:
    """
    find shortest path from start to finish
    :input:
        arg1: pred, predecessor list
        arg2: start
        arg3: finish
    :time complexity: O(|D|)
    :aux space complexity: O(|D|)
    """
    path = [finish]
    parent = pred[finish]
    while bool(parent) and parent != start:
        path.append(parent)
        parent = pred[parent]

    if parent != start:
        return None
    else:
        path.append(start)
        path.reverse()
        return path
        
def longest_path(g: List[List], topo_order: List) -> Tuple[List] or None:
    """
    
    :input:
        arg1: g, a graph represented in adjacency list
        arg2: topo_order
    :time complexity: O(|D|+|P|) = O(|D|)
    :aux space complexity:
    """
    dist = [-inf for _ in range(len(g))] #list that hold the longest path
    dist[0] = 0

    pred = [None for _ in range(len(g))]
    for u in topo_order: #O(|P|)
        for item in g[u]:
            v, w = item
            if w + dist[u] > dist[v]: #if new dist > curr dist
                dist[v] = w + dist[u] #update dist
                pred[v] = u #update predecessor
    return dist, pred


def topo_sort_dfs(g: List[List])-> List:
    """
    topological sort with depth first search
    :input:
        arg1: g, a graph represented adjacency list
    :time complexity: O(|D| + |P|) = O(|D|) 
    :aux space complexity: dominanted by order, O(|P|)
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
        arg2: u, source
        arg3: vst, visited list
        arg4: order, topological order list
    :time complexity: O(|D| + |P|) = O(|D|) 
    :aux space complexity: dominanted by order, O(|P|)
    """
    vst[u] = True
    for item in g[u]:
        v = item[0]
        if not vst[v]:
            topo_sort_dfs_aux(g, v, vst, order)
    order.append(u)

def graph(list: List):
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
    #import doctest
    #doctest.testmod(verbose=True)
    
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
    #downhillScores = [(0, 6, -500), (1, 4, 100), (1, 2, 300),
    #                  (6, 3, -100), (6, 1, 200), (3, 4, 400), (3, 1, 400),
    #                  (5, 6, 700), (5, 1, 1000), (4, 2, 100)]
    #print(optimalRoute(downhillScores, 6, 6))
    pass