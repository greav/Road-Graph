from lxml import etree
from time import time
from math import pi, log, tan
import svgwrite
import csv
import sys
import os

def transformCoordinates(lat, lon):
    radius = 1
    multiplier = 100000

    latRad = lat * pi / 180
    lonRad = lon * pi / 180
    x = radius * lonRad
    y = log(tan(pi / 4 + latRad / 2))

    x *= multiplier
    y *= multiplier

    return x, y


def parse(file):
    bounds = {}     # {'minlat': str, 'minlon': str, 'maxlat': str, 'maxlot': str}
    nodes = {}      # {'id': {'lat': str, 'lon': str, used: bool, ways: []}}
    ways = {}       # {'id': []}
    highways = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                'unclassified', 'residential', 'road']

    for _, element in etree.iterparse(file, tag=['bounds', 'node', 'way']):
        if element.tag == 'bounds':
            bounds['minlat'] = element.get('minlat')
            bounds['minlon'] = element.get('minlon')
            bounds['maxlat'] = element.get('maxlat')
            bounds['maxlot'] = element.get('maxlon')
        elif element.tag == 'node':
            nodes[element.get('id')] = {'lat': element.get('lat'), 'lon': element.get('lon'), 'used': False, 'ways': []}
        elif element.tag == 'way':
            wayID = element.get('id')
            lst_nodesID = []
            for child in element.iter('nd', 'tag'):
                if child.tag == 'nd':
                    childID = child.get('ref')
                    lst_nodesID.append(childID)
                elif child.tag == 'tag' and child.get('k') == 'highway' and child.get('v') in highways:
                    for nodeID in lst_nodesID:
                        if nodeID in nodes:
                            nodes[nodeID]['ways'].append(wayID)
                    ways[wayID] = lst_nodesID

        element.clear()

    return bounds, nodes, ways


def generateAdjList(nodes, ways):
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


def main():

    if len(sys.argv) == 1:
        print('There are no input data!')
        sys.exit()
    else:
        file = sys.argv[1]

    if not os.path.exists('result'):
        os.makedirs('result')

    parse_time = time()

    bounds, nodes, ways = parse(file)

    print('Parse time: ', time() - parse_time)
    print('Executing...')

    adj_list_time = time()

    adj_list = generateAdjList(nodes, ways)

    with open('result/adjacency_list.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node', 'Adjacent nodes'])
        for ID in adj_list:
            writer.writerow([ID] + [[int(node) for node in adj_list[ID]]])

    print('Adj list write time: ', time() - adj_list_time)
    print('Executing...')

    svg_time = time()

    svg_document = svgwrite.Drawing(filename='result/map.svg', size=('6000px', '2000px'))
    for wayID in ways:
        printed_nodes = []
        for nodeID in ways[wayID]:
            if nodeID in nodes:
                nodes[nodeID]['used'] = True
                printed_nodes.append((transformCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
                                                         float(nodes[nodeID]['lon']) - float(bounds['minlon']))))
        svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))

    # for nodeID in adj_list:
    #     if nodeID in nodes and nodes[nodeID]['used']:
    #         svg_document.add(svgwrite.shapes.Circle(
    #             center=(transformCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
    #                                          float(nodes[nodeID]['lon']) - float(bounds['minlon']))), r=1, fill='red'))
    svg_document.save()

    print('Svg write time: ', time() - svg_time)
    print('Executing...')

    adj_matrix_time = time()

    fieldnames = list(adj_list.keys())

    with open('result/adjacency_matrix.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[''] + fieldnames)

        writer.writeheader()
        for node in fieldnames:
            written_row = {neighbour: (1 if neighbour in adj_list[node] else 0) for neighbour in fieldnames}
            written_row[''] = node
            writer.writerow(written_row)

    print('Adj matrix time: ', time() - adj_matrix_time)


if __name__ == "__main__":
    start_time = time()
    print('Executing...')
    main()
    print('Finished!')
    print('Total time: ', time()-start_time)
