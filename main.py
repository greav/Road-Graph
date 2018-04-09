from lxml import etree
from time import time
from math import pi, sqrt, cos, sin, atan2, radians
import random
import svgwrite
import csv
import sys
import os
import haversine
import heapq
import collections
from queue import Queue

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class GraphOSM:
    def __init__(self):
        self.nodes = {}
        self.ways = {}
        self.bounds = {}
        self._neighbors = {}
        self.hospitals = []
        self.svg_document = None

    def parse(self, file):
        # bounds = {}     # {'minlat': str, 'minlon': str, 'maxlat': str, 'maxlon': str}
        # nodes = {}  # {'id': {'lat': str, 'lon': str, used: bool, ways: []}}
        # ways = {}  # {'id': []}
        # hospitals = []
        highways = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary',
                    'primary_link', 'secondary', 'secondary_link', 'tertiary',
                    'tertiary_link', 'unclassified', 'road', 'residential']

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

    def get_distance(self, origin, destination):
        # Haversine distance, in kilometers
        point1 = self.nodes[origin]['lat'], self.nodes[origin]['lon']
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

    # def dijkstra(self, start, goal):
    #     # origin = self.closest_node(start)
    #     origin = start
    #     # destination = self.closest_node(goal)
    #     destination = goal
    #     priority_queue = PriorityQueue()
    #     priority_queue.put(origin, 0)
    #
    #     came_from = {}
    #     came_from[origin] = None
    #     came_from[destination] = None
    #
    #     cost = collections.defaultdict(lambda: float('inf'))
    #     cost[origin] = 0
    #
    #     while not priority_queue.empty():
    #         current = priority_queue.get()
    #
    #         if current == destination:
    #             break
    #
    #         for next in self.neighbors(current):
    #             new_cost = cost[current] + self.get_distance(current, next)
    #             if new_cost < cost[next]:
    #                 cost[next] = new_cost
    #                 priority = new_cost
    #                 priority_queue.put(next, priority)
    #                 came_from[next] = current
    #
    #     if not came_from[destination]:
    #         return float('inf'), []
    #
    #     current = destination
    #     path = [current]
    #     while current != origin:
    #         current = came_from[current]
    #         path.append(current)
    #     path.reverse()
    #
    #     path = [start] + path + [goal]
    #     distance = cost[destination] # + self.get_distance(start, origin) + self.get_distance(destination, goal)
    #     return distance, path

    def dijkstra(self, start, goal):
        origin = start
        destination = goal
        priority_queue = []
        heapq.heappush(priority_queue, (0, origin))

        came_from = {}
        came_from[origin] = None
        came_from[destination] = None

        cost = collections.defaultdict(lambda: float('inf'))
        cost[origin] = 0

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
        while current != origin:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # path = [start] + path + [goal]
        distance = cost[destination] # + self.get_distance(start, origin) + self.get_distance(destination, goal)
        return distance, path


    def levit(self, start, goal):
        origin = start
        destination = goal
        cost = collections.defaultdict(lambda: float('inf'))
        cost[origin] = 0

        state = collections.defaultdict(lambda: 2)

        primary_queue = Queue()
        urgent_queue = Queue()

        primary_queue.put(origin)
        state[origin] = 1

        came_from = {}
        came_from[origin] = None
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
        while current != origin:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # path = [start] + path + [goal]
        distance = cost[destination] #+ self.get_distance(start, origin) + self.get_distance(destination, goal)
        return distance, path

    def heuristic(self, a, b, *, type_heuristic='Euclid'):
        (x1, y1) = self.to_equirect_project(self.nodes[a]['lat'], self.nodes[a]['lon'])
        (x2, y2) = self.to_equirect_project(self.nodes[b]['lat'], self.nodes[b]['lon'])

        if type_heuristic == 'Manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        elif type_heuristic == 'Euclid':
            return sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif type_heuristic == 'Chebyshev':
            return max(abs(x1 - x2), abs(y1 - y2))

    def a_star(self, start, goal, *, type_heuristic='Euclid'):
        # origin = self.closest_node(start)
        # destination = self.closest_node(goal)
        origin = start
        destination = goal

        priority_queue = PriorityQueue()
        priority_queue.put(origin, 0)

        came_from = {}
        came_from[origin] = None
        came_from[destination] = None

        cost = collections.defaultdict(lambda: float('inf'))
        cost[origin] = 0

        while not priority_queue.empty():
            current = priority_queue.get()

            if current == destination:
                break

            for next in self.neighbors(current):
                new_cost = cost[current] + self.get_distance(current, next)
                if new_cost < cost[next]:
                    cost[next] = new_cost
                    priority = new_cost + self.heuristic(next, destination, type_heuristic=type_heuristic)
                    priority_queue.put(next, priority)
                    came_from[next] = current

        if not came_from[destination]:
            return float('inf'), []

        current = destination
        path = [current]
        while current != origin:
            current = came_from[current]
            path.append(current)
        path.reverse()

        path = [start] + path + [goal]
        distance = cost[destination] + self.get_distance(start, origin) + self.get_distance(destination, goal)

        return distance, path

    def to_equirect_project(self, lat, lon):
        radius = 6371

        lat_rad = (self.bounds['maxlat'] - lat) * pi / 180
        lon_rad = (lon - self.bounds['minlon']) * pi / 180

        min_lat = self.bounds['maxlat'] * pi / 180

        x = lon_rad * cos(min_lat) * radius
        y = lat_rad * radius

        return x, y

    def transform_coordinates(self, lat, lon, *, multiplier=60):
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

    if len(sys.argv) == 1:
        # print('There are no input data!')
        # sys.exit()
        pass
    else:
        file = sys.argv[1]

    file = 'krasnodar.osm'

    city = GraphOSM()

    city.parse(file)
    city.generate_adjlist()
    city.create_svgmap()


    # lat, lon = coordinate_input((city.bounds['minlat'], city.bounds['maxlat']),
    #                             (city.bounds['minlon'], city.bounds['maxlon']))

    lat, lon = 45.1208, 38.9263

    origin = 0
    city.nodes[origin] = {'lat': lat, 'lon': lon}
    closest_node = city.closest_node(origin)
    city.connect(origin, closest_node)

    hospitals = city.hospitals[:10]
    for hospital in hospitals:
        # city.nodes[hospital]['used'] = True
        closest_node = city.closest_node(hospital)
        city.connect(hospital, closest_node)

    destination = hospitals[4]
    print(city.nodes[destination])

    dijkstra_routes = []
    a_star_routes = []
    levit_routes = []

    for destination in hospitals:
        start_time = time()
        dijkstra_distance, dijkstra_path = city.dijkstra(origin, destination)
        print('dijkstra time: ', time() - start_time)
        dijkstra_routes.append((dijkstra_distance, dijkstra_path))
        # a_star_distance, a_star_path = city.a_star(origin, destination, type_heuristic='Chebyshev')
        # a_star_routes.append((a_star_distance, a_star_path))
        start_time = time()
        levit_distance, levit_path = city.levit(origin, destination)
        print('levit time: ', time() - start_time)
        levit_routes.append((levit_distance, levit_path))
        print('dijkstra dist == levit dist ?', dijkstra_distance == levit_distance)
        print('dijkstra path == levit path ?', dijkstra_path == levit_path)



    dijkstra_routes.sort()
    # a_star_routes.sort()
    # levit_routes.sort()
    #
    # print('Dijkstra routes == A star routes?', dijkstra_routes == a_star_routes)
    # print('Dijkstra routes == Levit routes?', dijkstra_routes == levit_routes)
    # print('Levit routes == A star routes?', levit_routes == a_star_routes)

    with open('result/routes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Start', 'Destination', 'Route'])
        for _, route in dijkstra_routes:
            start_end = [(lat, lon), (city.nodes[route[-1]]['lat'], city.nodes[route[-1]]['lon'])]
            row = start_end + [[(city.nodes[node]['lat'], city.nodes[node]['lon']) for node in route]]
            writer.writerow(row)

    for route in dijkstra_routes[1:]:
        city.draw_route(route[1], color='red', stroke_width=1)

    city.draw_route(dijkstra_routes[0][1], color='green', stroke_width=2)

    city.svg_document.add(svgwrite.shapes.Circle(
                           center=(city.transform_coordinates(city.nodes[origin]['lat'],
                                                              city.nodes[origin]['lon'])), r=3, fill='blue'))

    for hospital in hospitals:
        city.svg_document.add(svgwrite.shapes.Circle(
                            center=(city.transform_coordinates(city.nodes[hospital]['lat'],
                                                               city.nodes[hospital]['lon'])), r=3, fill='red'))




    keys = [node for node in city._neighbors
            if haversine.haversine((lat, lon), (city.nodes[node]['lat'], city.nodes[node]['lon'])) >= 4]
    keys = random.sample(keys, 100)

    # for key in keys:
    #     city.svg_document.add(svgwrite.shapes.Circle(
    #         center=(city.transform_coordinates(city.nodes[key]['lat'],
    #                                            city.nodes[key]['lon'])), r=3, fill='orange'))

    dijk_times, lev_times, astar_times = [], [], []

    print('Number\tDistance\tDijkstra\tLevit\t\tAstar')

    for i, key in enumerate(keys):
        start_time = time()
        dijkstra_distance, dijkstra_path = city.dijkstra(origin, key)
        dijkstra_time = time() - start_time
        dijk_times.append(dijkstra_time)

        start_time = time()
        levit_distance, levit_path = city.levit(origin, key)
        levit_time = time() - start_time
        lev_times.append(levit_time)


        start_time = time()
        a_star_distance, a_star_path = city.a_star(origin, key, type_heuristic='Euclid')
        a_star_time = time() - start_time
        astar_times.append(a_star_time)

        print("{0}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}\t\t{4:.3f}".format(i + 1, dijkstra_distance, dijkstra_time, levit_time, a_star_time))

    print('Max dijkstra time:', max(dijk_times))
    print('Min dijkstra time:', min(dijk_times))

    print('Max levit time:', max(lev_times))
    print('Min levit time:', min(lev_times))

    print('Max astar time:', max(astar_times))
    print('Min astar time:', min(astar_times))

    # print('\nTesting heuristic functions in A*\n')
    # print('Number\tDistance\tEuclid\tDistance\tChebyshev\tDistance\tManhattan')
    #
    # for i, key in enumerate(keys):
    #     start_time = time()
    #     euclid_distance, a_star_path = city.a_star(origin, key, type_heuristic='Euclid')
    #     euclid_time = time() - start_time
    #
    #     start_time = time()
    #     chebyshev_distance, a_star_path = city.a_star(origin, key, type_heuristic='Chebyshev')
    #     chebyshev_time = time() - start_time
    #
    #     start_time = time()
    #     manhattan_distance, a_star_path = city.a_star(origin, key, type_heuristic='Manhattan')
    #     manhattan_time = time() - start_time
    #
    #     print("{0}\t\t{1:.3f}\t\t{2:.3f}\t{3:.3f}\t\t{4:.3f}\t\t{5:.3f}\t\t{6:.3f}".format(
    #      i + 1, euclid_distance, euclid_time, chebyshev_distance, chebyshev_time, manhattan_distance, manhattan_time))
    #
    #
    city.svg_document.save()


if __name__ == "__main__":
    start_time = time()
    print('Executing...')
    main()
    print('Finished!')
    print('Total time: ', time()-start_time)
