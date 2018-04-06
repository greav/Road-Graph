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


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class Graph:
    def __init__(self, nodes, adj_list):
        self._neighbors = adj_list
        self.nodes = nodes

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

    def dijkstra(self, start, goal):
        origin = self.closest_node(start)
        destination = self.closest_node(goal)

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
                    priority = new_cost
                    priority_queue.put(next, priority)
                    came_from[next] = current

        if not came_from[destination]:
            print('No solution')
            return float('inf'), None

        current = destination
        path = [current]
        while current != origin:
            current = came_from[current]
            path.append(current)
        path.reverse()

        path = [start] + path + [goal]
        distance = cost[destination] + self.get_distance(start, origin) + self.get_distance(destination, goal)

        return distance, path

    def levit(self, start, goal):
        origin = self.closest_node(start)
        destination = self.closest_node(goal)

        cost = collections.defaultdict(lambda: float('inf'))
        cost[origin] = 0

        state = collections.defaultdict(lambda: 2)
        state[origin] = 1

        queue = collections.deque()
        queue.append(origin)

        came_from = {}
        came_from[origin] = None
        came_from[destination] = None

        while queue:
            current = queue.popleft()
            state[current] = 0
            for next in self.neighbors(current):
                length = self.get_distance(current, next)
                #self.heuristic(current, next)
                if cost[next] > cost[current] + length:
                    cost[next] = cost[current] + length
                    if state[next] == 2:
                        queue.append(next)
                    elif state[next] == 0:
                        queue.appendleft(next)
                    came_from[next] = current
                    state[next] = 1

        if not came_from[destination]:
            print('No solution')
            return float('inf'), None

        current = destination
        path = [current]
        while current != origin:
            current = came_from[current]
            path.append(current)
        path.reverse()

        path = [start] + path + [goal]
        distance = cost[destination] + self.get_distance(start, origin) + self.get_distance(destination, goal)

        return distance, path

    def heuristic(self, a, b, *, type_heuristic='Euclid'):
        (x1, y1) = to_equirect_project(float(self.nodes[a]['lat']), float(self.nodes[a]['lon']))
        (x2, y2) = to_equirect_project(float(self.nodes[b]['lat']), float(self.nodes[b]['lon']))

        if type_heuristic == 'Manhattan':
            return abs(x1 - x2) + abs(y1 - y2)

        elif type_heuristic == 'Euclid':
            return sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif type_heuristic == 'Chebyshev':
            return max(abs(x1 - x2), abs(y1 - y2))

    def a_star(self, start, goal, *, type_heuristic='Euclid'):
        origin = self.closest_node(start)
        destination = self.closest_node(goal)

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
            print('No solution')
            return float('inf'), None

        current = destination
        path = [current]
        while current != origin:
            current = came_from[current]
            path.append(current)
        path.reverse()

        path = [start] + path + [goal]
        distance = cost[destination] + self.get_distance(start, origin) + self.get_distance(destination, goal)

        return distance, path


bounds = {}


def to_equirect_project(lat, lon):
    radius = 6371

    lat_rad = (float(bounds['maxlat']) - lat) * pi / 180
    lon_rad = (lon - float(bounds['minlon'])) * pi / 180

    min_lat = float(bounds['maxlat']) * pi / 180

    x = lon_rad * cos(min_lat) * radius
    y = lat_rad * radius

    return x, y


def transform_coordinates(lat, lon, *, multiplier=60):
    x, y = to_equirect_project(lat, lon)
    x *= multiplier
    y *= multiplier
    return x, y


def parse(file):
    # bounds = {}     # {'minlat': str, 'minlon': str, 'maxlat': str, 'maxlon': str}
    global bounds
    nodes = {}      # {'id': {'lat': str, 'lon': str, used: bool, ways: []}}
    ways = {}       # {'id': []}
    hospitals = []
    # highways = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    #             'unclassified', 'residential', 'road']
    highways = ["motorway", "motorway_link", "trunk", "trunk_link", "primary",
                "primary_link", "secondary", "secondary_link", "tertiary",
                "tertiary_link", "unclassified", "road", "residential", 'service']
    for _, element in etree.iterparse(file, tag=['bounds', 'node', 'way']):
        if element.tag == 'bounds':
            bounds['minlat'] = float(element.get('minlat'))
            bounds['minlon'] = float(element.get('minlon'))
            bounds['maxlat'] = float(element.get('maxlat'))
            bounds['maxlon'] = float(element.get('maxlon'))
        elif element.tag == 'node':
            nodeID = element.get('id')
            nodes[nodeID] = {'lat': float(element.get('lat')), 'lon': float(element.get('lon')),
                             'used': False, 'ways': []}
            for child in element.iter('tag'):
                value = child.get('v')
                if value == 'hospital':
                    hospitals.append(nodeID)
                    break
                else:
                    continue

        elif element.tag == 'way':
            wayID = element.get('id')
            lst_nodesID = []
            for child in element.iter('nd', 'tag'):
                if child.tag == 'nd':
                    childID = child.get('ref')
                    if childID not in nodes:
                        continue
                    lst_nodesID.append(childID)
                elif child.tag == 'tag' and child.get('k') == 'highway' and child.get('v') in highways:
                    for nodeID in lst_nodesID:
                        if nodeID in nodes:
                            nodes[nodeID]['ways'].append(wayID)
                    ways[wayID] = lst_nodesID

        element.clear()

    return bounds, nodes, ways, hospitals


def generate_short_adjList(nodes, ways):
    adj_list = {}

    for wayID in ways:
        for nodeID in ways[wayID]:
            # если узла нет в списке узлов, то переходим к следующему узлу
            if nodeID not in nodes:
                continue
            if len(nodes[nodeID]['ways']) == 1:
                index = ways[wayID].index(nodeID)
                if index == 0:
                    adj_list[nodeID] = {ways[wayID][-1]}

                elif index == (len(ways[wayID]) - 1):
                    adj_list[nodeID] = {ways[wayID][0]}
            elif len(nodes[nodeID]['ways']) > 1:
                for wayID2 in nodes[nodeID]['ways']:
                    if wayID2 != wayID:
                        index = ways[wayID2].index(nodeID)
                        # left_neighbour, right_neighbour = ways[wayID2][index-1:index], ways[wayID2][index+1:index+2]
                        for neighbour in ways[wayID2][index-1::-1]:
                            if neighbour in nodes and len(nodes[neighbour]['ways']) > 1:
                                if nodeID not in adj_list:
                                    adj_list[nodeID] = set()
                                    adj_list[nodeID].add(neighbour)
                                else:
                                    adj_list[nodeID].add(neighbour)
                                break
                            else:
                                continue

                        for neighbour in ways[wayID2][index + 1:]:
                            if neighbour in nodes and len(nodes[neighbour]['ways']) > 1:
                                if nodeID not in adj_list:
                                    adj_list[nodeID] = set()
                                    adj_list[nodeID].add(neighbour)
                                else:
                                    adj_list[nodeID].add(neighbour)
                                break
                            else:
                                continue

    return adj_list


def generate_adjlist(ways):
    adj_list = collections.defaultdict(set)

    for wayID in ways:

        if len(ways[wayID]) < 2:
            continue

        adj_list[ways[wayID][0]].add(ways[wayID][1])

        i = 1
        while i < len(ways[wayID]) - 1:
            adj_list[ways[wayID][i]].add(ways[wayID][i - 1])
            adj_list[ways[wayID][i]].add(ways[wayID][i + 1])
            i += 1

        adj_list[ways[wayID][i]].add(ways[wayID][i - 1])

    return adj_list


def main():

    if len(sys.argv) == 1:
        # print('There are no input data!')
        # sys.exit()
        pass
    else:
        file = sys.argv[1]

    if not os.path.exists('result'):
        os.makedirs('result')

    file = 'krasnodar.osm'


    parse_time = time()

    bounds, nodes, ways, hospitals = parse(file)




    print('Parse time: ', time() - parse_time)
    print('Executing...')

    #example 45.1359, 39.0088

    while True:
        lat = float(input('Enter latitude ({0} < lat < {1}): '.format(bounds['minlat'], bounds['maxlat'])))
        lon = float(input('Enter longitude ({0} < lon < {1}): '.format(bounds['minlon'], bounds['maxlon'])))
        if (lat < bounds['minlat'] or lat > bounds['maxlat'] or
            lon < bounds['minlon'] or lon > bounds['maxlon']):
            print('Wrong input!\nTry again', end='\n\n')
            continue
        else:
            break

    # lat, lon = 45.1359, 39.0088

    adj_list_time = time()

    adj_list = generate_adjlist(ways)

    with open('result/adjacency_list.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node', 'Adjacent nodes'])
        for ID in adj_list:
            writer.writerow([ID] + [[int(node) for node in adj_list[ID]]])

    print('Adj list write time: ', time() - adj_list_time)
    print('Executing...')

    svg_time = time()

    xpx, ypx = transform_coordinates(float(bounds['minlat']), float(bounds['maxlon']))

    svg_document = svgwrite.Drawing(filename='result/map.svg', size=(str(xpx), str(ypx)))




    for wayID in ways:
        printed_nodes = []
        for nodeID in ways[wayID]:
            if nodeID in nodes:
                nodes[nodeID]['used'] = True
                printed_nodes.append((transform_coordinates(float(nodes[nodeID]['lat']),
                                                         float(nodes[nodeID]['lon']))))
        svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))


    print('Svg write time: ', time() - svg_time)
    print('Executing...')

    # adj_matrix_time = time()
    #
    # fieldnames = list(adj_list.keys())
    #
    # with open('result/adjacency_matrix.csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=[''] + fieldnames)
    #
    #     writer.writeheader()
    #     for node in fieldnames:
    #         written_row = {neighbour: (1 if neighbour in adj_list[node] else 0) for neighbour in fieldnames}
    #         written_row[''] = node
    #         writer.writerow(written_row)
    #
    # print('Adj matrix time: ', time() - adj_matrix_time)



    city = Graph(nodes, adj_list)

    origin = '000000'
    city.nodes[origin] = {'lat': lat, 'lon': lon}

    hospitals = hospitals[:10]

    destination = hospitals[4]
    print(nodes[destination])

    dijkstra_routes = []
    a_star_routes = []
    levit_routes = []

    # for destination in hospitals:
    #     dijkstra_distance, dijkstra_path = city.dijkstra(origin, destination)
    #     dijkstra_routes.append((dijkstra_distance, dijkstra_path))
    #     a_star_distance, a_star_path = city.a_star(origin, destination, type_heuristic='Manhattan')
    #     a_star_routes.append((a_star_distance, a_star_path))
    #     # levit_distance, levit_path = city.dijkstra(origin, destination)
    #     # levit_routes.append((levit_distance, levit_path))



    # dijkstra_routes.sort()
    # a_star_routes.sort()

    # for route in dijkstra_routes:
    #     print(route)
    #
    # print('*'*20)
    #
    # for route in a_star_routes:
    #     print(route)
    #
    # print('*'*20)
    #
    #
    # #
    # # for i in range(len(dijkstra_routes)):
    # #     print(i + 1, '==>', dijkstra_routes == levit_routes)
    #
    # for i in range(len(dijkstra_routes)):
    #     print(i+1, '==>', dijkstra_routes[i] == a_star_routes[i])


    # for route in a_star_routes[1:]:
    #     printed_nodes = []
    #     for node in route[1]:
    #         printed_nodes.append((transform_coordinates(float(nodes[node]['lat']),
    #                                                    float(nodes[node]['lon']))))
    #     svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='red', stroke_width=1))
    #
    # printed_nodes = []
    # for node in a_star_routes[0][1]:
    #     printed_nodes.append((transform_coordinates(float(nodes[node]['lat']),
    #                                                float(nodes[node]['lon']))))
    # svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='green', stroke_width=2))
    #
    # svg_document.add(svgwrite.shapes.Circle(
    #                     center=(transform_coordinates(float(nodes[origin]['lat']),
    #                                                  float(nodes[origin]['lon']))), r=5, fill='blue'))
    #
    # for hospital in hospitals:
    #     svg_document.add(svgwrite.shapes.Circle(
    #                         center=(transform_coordinates(float(nodes[hospital]['lat']),
    #                                                      float(nodes[hospital]['lon']))), r=5, fill='red'))


    keys = random.sample(list(adj_list), 100)
    for key in keys:
        svg_document.add(svgwrite.shapes.Circle(
            center=(transform_coordinates(float(nodes[key]['lat']),
                                         float(nodes[key]['lon']))), r=3, fill='orange'))

    dijkstra_distance, dijkstra_path = city.dijkstra(origin, destination)
    a_star_distance, a_star_path = city.a_star(origin, destination, type_heuristic='Manhattan')
    levit_distance, levit_path = city.dijkstra(origin, destination)

    print('Dikstra == levit?', dijkstra_path == levit_path)
    print('Dikstra == astar?', dijkstra_path == a_star_path)
    print('Astar   == levit?', a_star_path == levit_path)

    print('Dikstra dist', dijkstra_distance)
    print('astar dist?', a_star_distance)
    print('levit dist?', levit_distance)

    print('Dikstra dist == levit dist?', dijkstra_distance == levit_distance)
    print('Dikstra dist == astar dist?', dijkstra_distance == a_star_distance)
    print('Astar dist == levit dist?', a_star_distance == levit_distance)

    printed_nodes = []
    for node in dijkstra_path:
        printed_nodes.append((transform_coordinates(float(nodes[node]['lat']),
                                                   float(nodes[node]['lon']))))
    svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='blue', stroke_width=1))

    printed_nodes2 = []
    for node in a_star_path:
        printed_nodes2.append((transform_coordinates(float(nodes[node]['lat']),
                                                    float(nodes[node]['lon']))))
    svg_document.add(svgwrite.shapes.Polyline(printed_nodes2, fill='none', stroke='green', stroke_width=2))


    svg_document.save()




if __name__ == "__main__":
    start_time = time()
    print('Executing...')
    main()
    print('Finished!')
    print('Total time: ', time()-start_time)
