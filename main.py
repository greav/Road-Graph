from lxml import etree
from time import time
from math import pi, sqrt, cos
from queue import Queue
import random
import svgwrite
import csv
import sys
import os
import haversine
import heapq
import collections
from collections import namedtuple



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

    # def get_closest(self, hospital, hospitals, visited):
    #     best_distance = float('inf')
    #
    #     # for h in map_hospitals:
    #     #
    #     #     if h not in visited:
    #     #         distance, _path = map_hospitals[hospital][h]
    #     #         if distance < best_distance:
    #     #             closest_hospital = h
    #     #             best_distance = distance
    #
    #
    #     for h in hospitals:
    #
    #         if h not in visited:
    #             distance = self.get_distance(hospital, h)
    #             if distance < best_distance:
    #                 closest_hospital = h
    #                 best_distance = distance
    #
    #     return closest_hospital, best_distance

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
            w = stack[-1]
            for edge in double_span_tree:
                if edge.start_node == w:
                    stack.append(edge.end_node)
                    double_span_tree.remove(edge)
                    break
            if w == stack[-1]:
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


class Logger():
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("result/testing.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()


def task2():
    main_time = time()
    print('Executing...')

    if len(sys.argv) == 1:
        print('There are no input data!')
        sys.exit()
    else:
        file = sys.argv[1]

    city = GraphOSM()

    city.parse(file)
    city.generate_adjlist()
    city.create_svgmap('map_with_routes.svg')

    lat, lon = coordinate_input((city.bounds['minlat'], city.bounds['maxlat']),
                                (city.bounds['minlon'], city.bounds['maxlon']))

    origin = 0
    city.nodes[origin] = {'lat': lat, 'lon': lon}
    closest_node = city.closest_node(origin)
    city.connect(origin, closest_node)

    hospitals = city.hospitals[:10]
    for hospital in hospitals:
        closest_node = city.closest_node(hospital)
        city.connect(hospital, closest_node)

    dijkstra_routes = []

    for destination in hospitals:
        dijkstra_distance, dijkstra_path = city.dijkstra(origin, destination)
        dijkstra_routes.append((dijkstra_distance, dijkstra_path))

    dijkstra_routes.sort()

    with open('result/routes_with_coords.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Start', 'Destination', 'Route'])
        for _, route in dijkstra_routes:
            start_end = [(lat, lon), (city.nodes[route[-1]]['lat'], city.nodes[route[-1]]['lon'])]
            row = start_end + [[(city.nodes[node]['lat'], city.nodes[node]['lon']) for node in route]]
            writer.writerow(row)

    with open('result/routes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for _, route in dijkstra_routes:
            writer.writerow(route)

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

    city.svg_document.save()

    starting_points = random.sample(list(city._neighbors), 100)
    dijkstra_total_time, levit_total_time, a_star_total_time = 0, 0, 0
    all_dijkstra_distances, all_levit_distances, all_astar_distances = [], [], []
    dijkstra_err, levit_err, a_euclid_err, a_manhattan_err, a_chebyshev_err = 0, 0, 0, 0, 0
    destination = hospitals[0]

    temp = sys.stdout
    sys.stdout = Logger()

    print('Number\tDistance\tDijkstra\tLevit\t\tAstar_Euclid\t\tAstar_Cheb\t\tAstar_Manhat')

    for i, point in enumerate(starting_points):
        start_time = time()
        dijkstra_distance, dijkstra_path = city.dijkstra(point, destination)
        dijkstra_time = time() - start_time
        dijkstra_total_time += dijkstra_time
        all_dijkstra_distances.append(dijkstra_distance)

        start_time = time()
        levit_distance, levit_path = city.levit(point, destination)
        levit_time = time() - start_time
        levit_total_time += levit_time
        all_levit_distances.append(levit_distance)

        start_time = time()
        a_star_distance, a_star_path = city.a_star(point, destination, type_heuristic='Euclid')
        a_star_time = time() - start_time
        a_star_total_time += a_star_time
        all_astar_distances.append(a_star_distance)

        start_time = time()
        a_star_cheb_distance, a_star_cheb_path = city.a_star(point, destination, type_heuristic='Chebyshev')
        a_star_cheb_time = time() - start_time

        start_time = time()
        a_star_manhat_distance, a_star_manhat_path = city.a_star(point, destination, type_heuristic='Manhattan')
        a_star_manhat_time = time() - start_time

        print("{0}\t\t{1:.3f}\t\t{2:.3f}\t\t{3:.3f}\t\t{4:.3f}\t\t\t\t{5:.3f}\t\t\t{6:.3f}".format(i + 1,
              dijkstra_distance, dijkstra_time, levit_time, a_star_time, a_star_cheb_time, a_star_manhat_time))

        exact_distance = min(dijkstra_distance, levit_distance)
        if exact_distance != float('inf'):
            dijkstra_err += dijkstra_distance/exact_distance - 1
            levit_err += levit_distance/exact_distance - 1
            a_euclid_err += a_star_distance/exact_distance - 1
            a_chebyshev_err += a_star_cheb_distance/exact_distance - 1
            a_manhattan_err += a_star_manhat_distance/exact_distance - 1

    print('\nDijkstra total time: {0:.3f} sec'.format(dijkstra_total_time))
    print('Levit total time: {0:.3f} sec'.format(levit_total_time))
    print('Astar total time: {0:.3f} sec'.format(a_star_total_time))

    try:
        all_dijkstra_distances.remove(float('inf'))
        all_levit_distances.remove(float('inf'))
        all_astar_distances.remove(float('inf'))
    except ValueError:
        pass

    n = len(all_dijkstra_distances)

    print('\nAverage dijkstra error: {0:.1f}%'.format(dijkstra_err * 100 / n))
    print('Average levit error: {0:.1f}%'.format(levit_err * 100 / n))
    print('Average astar (Euclid) error: {0:.1f}%'.format(a_euclid_err * 100 / n))
    print('Average astar (Chebyshev) error: {0:.1f}%'.format(a_chebyshev_err * 100 / n))
    print('Average astar (Manhattan) error: {0:.1f}%'.format(a_manhattan_err * 100 / n))

    print('\nDijkstra average arrival time: {0:.3f} min'.format(sum(all_dijkstra_distances)/len(all_dijkstra_distances)/40*60))
    print('Levit average arrival time: {0:.3f} min'.format(sum(all_levit_distances)/len(all_levit_distances)/40*60))
    print('Astar average arrival time: {0:.3f} min'.format(sum(all_astar_distances)/len(all_astar_distances)/40*60))

    sys.stdout.close()
    sys.stdout = temp

    print('Finished!')
    print('Total time: {0:.3f}'.format(time() - main_time))


def main():
    main_time = time()
    print('Executing...')

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

    # lat, lon = coordinate_input((city.bounds['minlat'], city.bounds['maxlat']),
    #                             (city.bounds['minlon'], city.bounds['maxlon']))

    lat, lon = 45.1166, 38.9885
    # lat, lon = 45.0278, 39.0743

    origin = 0
    city.nodes[origin] = {'lat': lat, 'lon': lon}
    closest_node = city.closest_node(origin)
    city.connect(origin, closest_node)

    hospitals = city.hospitals[:6] + city.hospitals[7:11]
    for hospital in hospitals:
        closest_node = city.closest_node(hospital)
        city.connect(hospital, closest_node)

    tsp_edge_list, tsp_path_matrix = city.generate_tsp_distance_matrix([origin] + hospitals)

    cycle, length = city.tsp_nearest_neighbour(tsp_path_matrix, [origin] + hospitals)

    print('cycle:')
    print(cycle)
    print('NNA distance: ', length)

    printed_routes = []
    i = 0
    while i < len(cycle) - 1:
        _distance, path = tsp_path_matrix[cycle[i]][cycle[i + 1]]
        printed_routes.append(path)
        i += 1


    with open('result/nna_routes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for route in printed_routes:
            writer.writerow(route)

    city.create_svgmap('svgmap_tsp_nna2.svg')

    for route in printed_routes:
        city.draw_route(route, color='red', stroke_width=1)


    for i, hospital in enumerate(cycle[:-1], start=1):
        city.svg_document.add(svgwrite.shapes.Circle(
            center=(city.transform_coordinates(city.nodes[hospital]['lat'],
                                               city.nodes[hospital]['lon'])), r=3, fill='red'))
        city.svg_document.add(city.svg_document.text(i,
                                                     insert=(city.transform_coordinates(city.nodes[hospital]['lat'],
                                                                                        city.nodes[hospital]['lon'])), ))
    city.svg_document.save()


    cycle, length2 = city.tsp_double_min_spanning_tree(origin, hospitals, tsp_path_matrix, tsp_edge_list)

    print(cycle)
    print('DMTS distance:', length2)

    printed_routes = []
    i = 0
    while i < len(cycle) - 1:
        _distance, path = tsp_path_matrix[cycle[i]][cycle[i + 1]]
        printed_routes.append(path)
        i += 1

    with open('result/dmst_routes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for route in printed_routes:
            writer.writerow(route)

    city.create_svgmap('svgmap_dmts_dmts2.svg')

    for route in printed_routes:
        city.draw_route(route, color='red', stroke_width=1)

    for i, hospital in enumerate(cycle[:-1], start=1):
        city.svg_document.add(svgwrite.shapes.Circle(
            center=(city.transform_coordinates(city.nodes[hospital]['lat'],
                                               city.nodes[hospital]['lon'])), r=3, fill='red'))
        city.svg_document.add(city.svg_document.text(i,
                                                     insert=(city.transform_coordinates(city.nodes[hospital]['lat'],
                                                                                        city.nodes[hospital][
                                                                                            'lon'])), ))
    city.svg_document.save()

    print('Finished!')
    print('Total time: {0:.3f}'.format(time() - main_time))


def main2():
    main_time = time()
    print('Executing...')
    file = 'krasnodar.osm'

    city = GraphOSM()

    city.parse(file)
    city.generate_adjlist()

    # lat, lon = coordinate_input((city.bounds['minlat'], city.bounds['maxlat']),
    #                             (city.bounds['minlon'], city.bounds['maxlon']))

    lat, lon = 45.1166, 38.9885

    origin = 0
    city.nodes[origin] = {'lat': lat, 'lon': lon}
    closest_node = city.closest_node(origin)
    city.connect(origin, closest_node)

    hospitals = city.hospitals[:6] + city.hospitals[7:11]
    for hospital in hospitals:
        closest_node = city.closest_node(hospital)
        city.connect(hospital, closest_node)


    tsp_edge_list, tsp_path_matrix = city.generate_tsp_distance_matrix([origin] + hospitals)
    order, length = city.tsp_double_min_spanning_tree(origin, hospitals, tsp_path_matrix)
    print(order)

    printed_routes = []
    i = 0
    while i < len(order) - 1:
        _distance, path = city.a_star(order[i], order[i + 1])
        printed_routes.append(path)
        i += 1


if __name__ == "__main__":
    main()

