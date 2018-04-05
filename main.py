from lxml import etree
from time import time
from math import pi, log, tan, sqrt, sin, cos
import svgwrite
import csv
import sys
import os
import haversine
import heapq
import collections


Route = collections.namedtuple('Route', 'distance path')


class Heap(object):
    def __init__(self):
        self._values = []

    def push(self, value):
        heapq.heappush(self._values, value)

    def pop(self):
        return heapq.heappop(self._values)

    def __len__(self):
        return len(self._values)


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
        point1 = float(self.nodes[origin]['lat']), float(self.nodes[origin]['lon'])
        point2 = float(self.nodes[destination]['lat']), float(self.nodes[destination]['lon'])
        distance = haversine.haversine(point1, point2)
        return distance


    def closestNode(self, destination):

        min_node, attributes = self._neighbors.popitem()
        min_distance = self.get_distance(min_node, destination)
        self._neighbors[min_node] = attributes

        for node in self._neighbors:
            distance = self.get_distance(node, destination)
            if distance < min_distance:
                min_distance = distance
                min_node = node

        return min_node

    def dijkstra(self, origin, destination):

        destination = self.closestNode(destination)

        routes = Heap()
        for neighbor in self.neighbors(origin):
            distance = self.get_distance(origin, neighbor)
            routes.push(Route(distance=distance, path=[origin, neighbor]))

        visited = set()
        visited.add(origin)

        while routes:

            distance, path = routes.pop()
            node = path[-1]
            if node in visited:
                continue

            # We have arrived! Wo-hoo!
            if node == destination:
                return distance, path

            # Tentative distances to all the unvisited neighbors
            for neighbor in self.neighbors(node):
                if neighbor not in visited and neighbor in self.nodes:
                    # Total spent so far plus the price of getting there
                    new_distance = distance + self.get_distance(node, neighbor)
                    new_path = path + [neighbor]
                    routes.push(Route(new_distance, new_path))

            visited.add(node)

        return float('infinity')

    def levit(self, origin, destination):

        destination = self.closestNode(destination)

        d = {nodeID: float('inf') for nodeID in self._neighbors}
        d[origin] = 0

        state = {nodeID: 2 for nodeID in self._neighbors}
        state[origin] = 1

        q = collections.deque()
        q.append(origin)

        p = {nodeID: -1 for nodeID in self._neighbors}

        while q:
            vertex = q.popleft()
            state[vertex] = 0
            for to in self._neighbors[vertex]:
                length = self.get_distance(vertex, to)
                if d[to] > (d[vertex] + length):
                    d[to] = d[vertex] + length
                    if state[to] == 2:
                        q.append(to)
                    elif state[to] == 0:
                        q.appendleft(to)
                    p[to] = vertex
                    state[to] = 1

        if p[destination] == -1:
            print('No solution')
            return float('inf'), -1
        else:
            path = []
            vertex = destination

            while vertex != -1:
                path.append(vertex)
                vertex = p[vertex]

            path.reverse()

            return d[destination], path

    def heuristic(self, a, b, *, type_h):
        (x1, y1) = float(self.nodes[a]['lat']), float(self.nodes[a]['lon'])
        (x2, y2) = float(self.nodes[b]['lat']), float(self.nodes[b]['lon'])

        if type_h == 'Manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        elif type_h == 'Euclid':
            return sqrt((x1 - x2)**2 + (y1 - y2)**2)
        elif type_h == 'Chebyshev':
            return max(abs(x1 - x2), abs(y1 - y2))

    def astar(self, start, goal, *, type_heuristic='Manhattan'):
        goal = self.closestNode(goal)

        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + self.get_distance(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next, type_h=type_heuristic)
                    frontier.put(next, priority)
                    came_from[next] = current

        current = goal
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()

        return cost_so_far[goal], path



# def transformCoordinates(lat, lon):
#     radius = 6378
#     multiplier = 30
#
#
#
#     latRad = lat * pi / 180
#     lonRad = lon * pi / 180
#     x = radius * lonRad
#     y = radius*log(tan(pi / 4 + latRad / 2))
#
#     x *= multiplier
#     y *= multiplier
#
#     return x, y

bounds = {}

def transformCoordinates(lat, lon):
    radius = 6378
    multiplier = 23



    latRad = (lat - float(bounds['minlat'])) * pi / 180
    lonRad = (lon - float(bounds['minlon'])) * pi / 180

    minlat = float(bounds['minlat']) * pi / 180

    x = lonRad * cos(minlat) * radius
    y = latRad * radius

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
            bounds['minlat'] = element.get('minlat')
            bounds['minlon'] = element.get('minlon')
            bounds['maxlat'] = element.get('maxlat')
            bounds['maxlon'] = element.get('maxlon')
        elif element.tag == 'node':
            nodeID = element.get('id')
            nodes[nodeID] = {'lat': element.get('lat'), 'lon': element.get('lon'),
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


def generateShortAdjList(nodes, ways):
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


def generateAdjList(ways):
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

    adj_list_time = time()

    # adj_list = generateShortAdjList(nodes, ways)
    adj_list = generateAdjList(ways)

    with open('result/adjacency_list.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node', 'Adjacent nodes'])
        for ID in adj_list:
            writer.writerow([ID] + [[int(node) for node in adj_list[ID]]])

    print('Adj list write time: ', time() - adj_list_time)
    print('Executing...')

    svg_time = time()

    xpx, ypx = transformCoordinates(float(bounds['maxlat']),
                         float(bounds['maxlon']))

    svg_document = svgwrite.Drawing(filename='result/map.svg', size=(str(xpx), str(ypx)))




    for wayID in ways:
        printed_nodes = []
        for nodeID in ways[wayID]:
            if nodeID in nodes:
                nodes[nodeID]['used'] = True
                printed_nodes.append((transformCoordinates(float(nodes[nodeID]['lat']) - float(bounds['maxlat']),
                                                         float(nodes[nodeID]['lon']) - float(bounds['minlon']))))
        svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))

    # for nodeID in adj_list:
    #     if nodeID in nodes and nodes[nodeID]['used']:
    #         svg_document.add(svgwrite.shapes.Circle(
    #             center=(transformCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
    #                                          float(nodes[nodeID]['lon']) - float(bounds['minlon']))), r=1, fill='red'))



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

    origin = '2931369372'

    dijkstra_distance, dijkstra_path = city.dijkstra(origin, hospitals[2])
    print(dijkstra_distance)
    # print(dijkstra_path)

    levit_distance, levit_path = city.levit(origin, hospitals[2])
    print(levit_distance)
    # print(levit_path)

    astar_distance, astar_path = city.astar(origin, hospitals[2], type_heuristic='Chebyshev')
    print(astar_distance)
    # print(astar_path)

    print(dijkstra_path == levit_path)
    print(astar_path == dijkstra_path)




    printed_nodes = []

    for nodeID in dijkstra_path:
        printed_nodes.append((transformCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
                                                   float(nodes[nodeID]['lon']) - float(bounds['minlon']))))
    svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='red', stroke_width=1))

    for hospital in hospitals[2:3]:
        svg_document.add(svgwrite.shapes.Circle(
                        center=(transformCoordinates(float(nodes[hospital]['lat']) - float(bounds['minlat']),
                                                     float(nodes[hospital]['lon']) - float(bounds['minlon']))), r=1.5, fill='red'))


    svg_document.add(svgwrite.shapes.Circle(
        center=(transformCoordinates(float(nodes[origin]['lat']) - float(bounds['minlat']),
                                     float(nodes[origin]['lon']) - float(bounds['minlon']))), r=1.5, fill='green'))


    minlat = float('inf')
    minlon = float('inf')
    maxlat = float('-inf')
    maxlon = float('-inf')

    for wayID in ways:
        for nodeID in ways[wayID]:
            if float(nodes[nodeID]['lat']) < minlat:
                minlat = float(nodes[nodeID]['lat'])
            if float(nodes[nodeID]['lon']) < minlon:
                minlon = float(nodes[nodeID]['lon'])

            if float(nodes[nodeID]['lat']) > maxlat:
                maxlat = float(nodes[nodeID]['lat'])
            if float(nodes[nodeID]['lon']) > maxlon:
                maxlon = float(nodes[nodeID]['lon'])

    svg_document.add(svgwrite.shapes.Circle(
        center=(transformCoordinates(float(minlat) - float(bounds['minlat']),
                                     float(minlon) - float(bounds['minlon']))), r=3, fill='green'))

    svg_document.add(svgwrite.shapes.Circle(
        center=(transformCoordinates(float(maxlat) - float(bounds['minlat']),
                                     float(maxlon) - float(bounds['minlon']))), r=3, fill='green'))

    print('min lat = ', minlat, 'min lon = ', minlon)
    print(bounds['minlat'], bounds['minlon'])
    print('max lat = ', maxlat, 'max lon = ', maxlon)
    print(bounds['maxlat'], bounds['maxlon'])


    svg_document.save()

    print(city.get_distance(origin, hospitals[3]))

    x1, y1 = transformCoordinates(float(nodes[origin]['lat']) - float(bounds['minlat']),
                                                   float(nodes[origin]['lon']) - float(bounds['minlon']))

    x2, y2 = transformCoordinates(float(nodes[hospitals[3]]['lat']) - float(bounds['minlat']),
                                  float(nodes[hospitals[3]]['lon']) - float(bounds['minlon']))

    print(sqrt((x1-x2)**2 + (y1-y2)**2))




if __name__ == "__main__":
    start_time = time()
    print('Executing...')
    main()
    print('Finished!')
    print('Total time: ', time()-start_time)
