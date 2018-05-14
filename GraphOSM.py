import csv
import svgwrite
import os
import haversine
import heapq
import collections
from collections import namedtuple
from lxml import etree
from math import pi, sqrt, cos
from queue import Queue


Edge = namedtuple('Edge', ['start_node', 'end_node', 'weight'])


class GraphOSM:
    def __init__(self):
        """
        bounds = {'minlat': str, 'minlon': str, 'maxlat': str, 'maxlon': str}
        nodes = {'id': {'lat': str, 'lon': str, used: bool, ways: []}}
        ways = {'id': []}
        """
        self.nodes = {}
        self.ways = {}
        self.bounds = {}
        self._neighbors = {}
        self.hospitals = []
        self.svg_document = None

    def parse(self, file):
        highways = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary',
                    'primary_link', 'secondary', 'secondary_link', 'tertiary',
                    'tertiary_link', 'unclassified', 'road', 'residential', 'service']

        for _, element in etree.iterparse(file, tag=['bounds', 'node', 'way']):
            if element.tag == 'bounds':
                self.bounds['minlat'] = float(element.get('minlat'))
                self.bounds['minlon'] = float(element.get('minlon'))
                self.bounds['maxlat'] = float(element.get('maxlat'))
                self.bounds['maxlon'] = float(element.get('maxlon'))
            elif element.tag == 'node':
                nodeID = element.get('id')
                self.nodes[nodeID] = {'lat': float(element.get('lat')), 'lon': float(element.get('lon')),
                                 'used': False, 'ways': []}
                for child in element.iter('tag'):
                    value = child.get('v')
                    if value == 'hospital':
                        self.hospitals.append(nodeID)
                        break
                    else:
                        continue

            elif element.tag == 'way':
                wayID = element.get('id')
                lst_nodesID = []
                for child in element.iter('nd', 'tag'):
                    if child.tag == 'nd':
                        childID = child.get('ref')
                        if childID not in self.nodes:
                            continue
                        lst_nodesID.append(childID)
                    elif child.tag == 'tag' and child.get('k') == 'highway' and child.get('v') in highways:
                        for nodeID in lst_nodesID:
                            if nodeID in self.nodes:
                                self.nodes[nodeID]['ways'].append(wayID)
                        self.ways[wayID] = lst_nodesID

            element.clear()

    def generate_short_adjList(self):
        adj_list = {}

        for wayID in self.ways:
            for nodeID in self.ways[wayID]:
                # если узла нет в списке узлов, то переходим к следующему узлу
                if nodeID not in self.nodes:
                    continue
                if len(self.nodes[nodeID]['ways']) == 1:
                    index = self.ways[wayID].index(nodeID)
                    if index == 0:
                        adj_list[nodeID] = {self.ways[wayID][-1]}

                    elif index == (len(self.ways[wayID]) - 1):
                        adj_list[nodeID] = {self.ways[wayID][0]}
                elif len(self.nodes[nodeID]['ways']) > 1:
                    for wayID2 in self.nodes[nodeID]['ways']:
                        if wayID2 != wayID:
                            index = self.ways[wayID2].index(nodeID)
                            # left_neighbour, right_neighbour = ways[wayID2][index-1:index], ways[wayID2][index+1:index+2]
                            for neighbour in self.ways[wayID2][index - 1::-1]:
                                if neighbour in self.nodes and len(self.nodes[neighbour]['ways']) > 1:
                                    if nodeID not in adj_list:
                                        adj_list[nodeID] = set()
                                        adj_list[nodeID].add(neighbour)
                                    else:
                                        adj_list[nodeID].add(neighbour)
                                    break
                                else:
                                    continue

                            for neighbour in self.ways[wayID2][index + 1:]:
                                if neighbour in self.nodes and len(self.nodes[neighbour]['ways']) > 1:
                                    if nodeID not in adj_list:
                                        adj_list[nodeID] = set()
                                        adj_list[nodeID].add(neighbour)
                                    else:
                                        adj_list[nodeID].add(neighbour)
                                    break
                                else:
                                    continue

        self._neighbors = adj_list

    def generate_adjlist(self):
        adj_list = collections.defaultdict(set)

        for wayID in self.ways:

            if len(self.ways[wayID]) < 2:
                continue

            adj_list[self.ways[wayID][0]].add(self.ways[wayID][1])

            i = 1
            while i < len(self.ways[wayID]) - 1:
                adj_list[self.ways[wayID][i]].add(self.ways[wayID][i - 1])
                adj_list[self.ways[wayID][i]].add(self.ways[wayID][i + 1])
                i += 1

            adj_list[self.ways[wayID][i]].add(self.ways[wayID][i - 1])

        self._neighbors = adj_list

    def adjlist_to_csv(self, filename='adjacency_list.csv'):
        if not os.path.exists('result'):
            os.makedirs('result')
        with open('result/' + filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Node', 'Adjacent nodes'])
            for ID in self._neighbors:
                writer.writerow([ID] + [[int(node) for node in self._neighbors[ID]]])

    def adjmatrix_to_csv(self, filename='adjacency_matrix.csv'):
        if not os.path.exists('result'):
            os.makedirs('result')

        fieldnames = list(self._neighbors.keys())

        with open('result/' + filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[''] + fieldnames)

            writer.writeheader()
            for node in fieldnames:
                write_line = {neighbour: (1 if neighbour in self._neighbors[node] else 0) for neighbour in fieldnames}
                write_line[''] = node
                writer.writerow(write_line)

    def create_svgmap(self, filename='map.svg'):
        if not os.path.exists('result'):
            os.makedirs('result')

        x_px, y_px = self.transform_coordinates(self.bounds['minlat'], self.bounds['maxlon'])
        self.svg_document = svgwrite.Drawing(filename='result/' + filename, size=(str(x_px), str(y_px)))

        for wayID in self.ways:
            printed_nodes = []
            for nodeID in self.ways[wayID]:
                if nodeID in self.nodes:
                    self.nodes[nodeID]['used'] = True
                    printed_nodes.append((self.transform_coordinates(self.nodes[nodeID]['lat'],
                                                                     self.nodes[nodeID]['lon'])))
            self.svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))

    def connect(self, node1, node2):
        self._neighbors[node1].add(node2)
        self._neighbors[node2].add(node1)

    def neighbors(self, node):
        yield from self._neighbors[node]

    def get_distance(self, start, destination):
        # Haversine distance, in kilometers
        point1 = self.nodes[start]['lat'], self.nodes[start]['lon']
        point2 = self.nodes[destination]['lat'], self.nodes[destination]['lon']
        distance = haversine.haversine(point1, point2)
        return distance

    def closest_node(self, destination):
        min_node_id, attributes = self._neighbors.popitem()
        min_distance = self.get_distance(min_node_id, destination)
        self._neighbors[min_node_id] = attributes

        for node in self._neighbors:
            distance = self.get_distance(node, destination)
            if distance < min_distance:
                min_distance = distance
                min_node_id = node

        return min_node_id

    def dijkstra(self, start, destination):
        priority_queue = []
        heapq.heappush(priority_queue, (0, start))

        came_from = {}
        came_from[start] = None
        came_from[destination] = None

        cost = collections.defaultdict(lambda: float('inf'))
        cost[start] = 0

        while priority_queue:
            current = heapq.heappop(priority_queue)[1]

            if current == destination:
                break

            for next in self.neighbors(current):
                new_cost = cost[current] + self.get_distance(current, next)
                if new_cost < cost[next]:
                    cost[next] = new_cost
                    priority = new_cost
                    heapq.heappush(priority_queue, (priority, next))
                    came_from[next] = current

        if not came_from[destination]:
            return float('inf'), []

        current = destination
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()

        return cost[destination], path

    def levit(self, start, destination):
        cost = collections.defaultdict(lambda: float('inf'))
        cost[start] = 0

        state = collections.defaultdict(lambda: 2)

        primary_queue = Queue()
        urgent_queue = Queue()

        primary_queue.put(start)
        state[start] = 1

        came_from = {}
        came_from[start] = None
        came_from[destination] = None

        while not primary_queue.empty() or not urgent_queue.empty():
            current = primary_queue.get() if urgent_queue.empty() else urgent_queue.get()
            state[current] = 0
            for next in self.neighbors(current):
                length = self.get_distance(current, next)
                if state[next] == 2:
                    primary_queue.put(next)
                    state[next] = 1
                    if cost[current] + length < cost[next]:
                        came_from[next] = current
                        cost[next] = cost[current] + length
                elif state[next] == 1 and cost[current] + length < cost[next]:
                        came_from[next] = current
                        cost[next] = cost[current] + length
                elif state[next] == 0 and cost[next] > cost[current] + length:
                    urgent_queue.put(next)
                    state[next] = 1
                    cost[next] = cost[current] + length
                    came_from[next] = current

        if not came_from[destination]:
            return float('inf'), []

        current = destination
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()

        return cost[destination], path

    def heuristic(self, a, b, *, type_heuristic='Euclid'):
        (x1, y1) = self.to_equirect_project(self.nodes[a]['lat'], self.nodes[a]['lon'])
        (x2, y2) = self.to_equirect_project(self.nodes[b]['lat'], self.nodes[b]['lon'])

        if type_heuristic == 'Manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        elif type_heuristic == 'Euclid':
            return sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif type_heuristic == 'Chebyshev':
            return max(abs(x1 - x2), abs(y1 - y2))

    def a_star(self, start, destination, *, type_heuristic='Euclid'):
        priority_queue = []
        heapq.heappush(priority_queue, (0, start))

        came_from = {}
        came_from[start] = None
        came_from[destination] = None

        cost = collections.defaultdict(lambda: float('inf'))
        cost[start] = 0

        while priority_queue:
            current = heapq.heappop(priority_queue)[1]

            if current == destination:
                break

            for next in self.neighbors(current):
                new_cost = cost[current] + self.get_distance(current, next)
                if new_cost < cost[next]:
                    cost[next] = new_cost
                    priority = new_cost + self.heuristic(next, destination, type_heuristic=type_heuristic)
                    heapq.heappush(priority_queue, (priority, next))
                    came_from[next] = current

        if not came_from[destination]:
            return float('inf'), []

        current = destination
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()

        return cost[destination], path

    def to_equirect_project(self, lat, lon):
        radius = 6371

        lat_rad = (self.bounds['maxlat'] - lat) * pi / 180
        lon_rad = (lon - self.bounds['minlon']) * pi / 180

        min_lat = self.bounds['maxlat'] * pi / 180

        x = lon_rad * cos(min_lat) * radius
        y = lat_rad * radius

        return x, y

    def transform_coordinates(self, lat, lon, *, multiplier=40):
        x, y = self.to_equirect_project(lat, lon)
        x *= multiplier
        y *= multiplier
        return x, y

    def draw_route(self, route, *, color='black', stroke_width=1):
        printed_nodes = []
        for node in route:
            printed_nodes.append((self.transform_coordinates(self.nodes[node]['lat'],
                                                             self.nodes[node]['lon'])))
        self.svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke=color, stroke_width=stroke_width))

    def get_closest(self, hospital, map_hospitals, visited):
        best_distance = float('inf')

        for h in map_hospitals:

            if h not in visited:
                distance, _path = map_hospitals[hospital][h]
                if distance < best_distance:
                    closest_hospital = h
                    best_distance = distance

        return closest_hospital, best_distance

    def tsp_nearest_neighbour(self, map_hospitals, hospitals):
        order = []
        order.append(hospitals[0])

        length = 0

        next_, dist = self.get_closest(order[0], map_hospitals, order)
        length += dist
        order.append(next_)

        while len(order) < len(map_hospitals):
            next_, dist = self.get_closest(next_, map_hospitals, order)
            length += dist
            order.append(next_)

        order.append(order[0])
        distance, _path = map_hospitals[order[-2]][order[-1]]
        length += distance

        return order, length

    def generate_tsp_distance_matrix(self, nodes):
        edge_list = []
        path_matrix = {}

        for start in nodes:
            path_matrix[start] = {}
            for end in nodes:
                if start != end:
                    distance, path = self.a_star(start, end)
                    edge_list.append(Edge(start, end, distance))
                    path_matrix[start][end] = distance, path
        return edge_list, path_matrix

    def min_spanning_tree(self, root, hospitals, edges):
        result = []
        edges.sort(key=lambda edge: edge.weight)

        selected_nodes = [root]
        while len(result) != len(hospitals):
            for edge in edges:
                if edge.start_node in selected_nodes and edge.end_node not in selected_nodes:
                    result.append(edge)
                    selected_nodes.append(edge.end_node)
                    break

        return result

    def tsp_double_min_spanning_tree(self, root, hospitals, path_matrix, edges):
        path = [root]
        length = 0
        double_span_tree = []
        for edge in self.min_spanning_tree(root, hospitals, edges):
            double_span_tree.append(edge)
            reverse_edge = Edge(edge.end_node, edge.start_node, edge.weight)
            double_span_tree.append(reverse_edge)

        stack = [root]
        while stack:
            cur_node = stack[-1]
            for edge in double_span_tree:
                if edge.start_node == cur_node:
                    stack.append(edge.end_node)
                    double_span_tree.remove(edge)
                    break
            if cur_node == stack[-1]:
                node = stack.pop()
                if node not in path:
                    path.append(node)

        path.append(path[0])
        i = 0
        while i < len(path) - 1:
            distance, _path = path_matrix[path[i]][path[i + 1]]
            length += distance
            i += 1

        return path, length


def coordinate_input(lat_range=(-90, 90), lon_range=(-180, 180)):
    min_lat, max_lat = lat_range
    min_lon, max_lon = lon_range
    while True:
        lat = float(input('Enter latitude ({0} < lat < {1}): '.format(min_lat, max_lat)))
        lon = float(input('Enter longitude ({0} < lon < {1}): '.format(min_lon, max_lon)))
        if (lat < min_lat or lat > max_lat or
            lon < min_lon or lon > max_lon):
            print('Wrong input!\nTry again', end='\n\n')
            continue
        else:
            return lat, lon


def main():
    print('Nothing to run')


if __name__ == "__main__":
    main()

