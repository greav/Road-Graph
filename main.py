from lxml import etree
from time import time
import svgwrite
import csv
import sys
import math
import os

def convertCoordinates(lat, lon):
    radius = 1
    multiplier = 500000

    latRad = lat * math.pi / 180
    lonRad = lon * math.pi / 180
    x = radius * lonRad
    y = math.log(math.tan(math.pi / 4 + latRad / 2))

    x *= multiplier
    y *= multiplier
    return x, y


def parse(file):
    bounds = {}     # {'minlat': str, 'minlon': str, 'maxlat': str, 'maxlot': str}
    nodes = {}      # {'id': {'lat': str, 'lon': str, used: bool}}
    ways = []       # [[], ... ,[]]
    # highways = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    #             'unclassified', 'residential', 'service', 'road']
    highways = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                'unclassified', 'residential', 'road']

    for _, element in etree.iterparse(file, tag=['bounds', 'node', 'way']):
        if element.tag == 'bounds':
            bounds['minlat'] = element.get('minlat')
            bounds['minlon'] = element.get('minlon')
            bounds['maxlat'] = element.get('maxlat')
            bounds['maxlot'] = element.get('maxlon')
        elif element.tag == 'node':
            nodes[element.get('id')] = {'lat': element.get('lat'), 'lon': element.get('lon'), 'used': False}
        elif element.tag == 'way':
            lst_nodesID = []
            for child in element.iter('nd', 'tag'):
                if child.tag == 'nd':
                    lst_nodesID.append(child.get('ref'))
                elif child.tag == 'tag' and child.get('k') == 'highway' and child.get('v') in highways:
                    ways.append(lst_nodesID)
        element.clear()

    return bounds, nodes, ways


def write_adj_list(nodes, ways):
    adj_list = {}
    for way1 in ways:
        for nodeID in way1:
            if nodeID not in nodes:
                continue
            for way2 in ways:
                if way1 != way2 and nodeID in way2:
                    index = way2.index(nodeID)
                    left_neighbour, right_neighbour = way2[index-1:index], way2[index+1:index+2]
                    if nodeID not in adj_list:
                        adj_list[nodeID] = set()
                        adj_list[nodeID].update(left_neighbour + right_neighbour)
                    else:
                        adj_list[nodeID].update(left_neighbour + right_neighbour)

    with open('output/adjacency_list.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for ID in adj_list:
            writer.writerow([ID] + list(adj_list[ID]))

    return adj_list


def main():

    if len(sys.argv) == 1:
        file = '/home/victor/osmosis/krasnodar.osm'
    else:
        file = sys.argv[1]

    if not os.path.exists('output'):
        os.makedirs('output')

    parse_time = time()

    bounds, nodes, ways = parse(file)

    print('Parse time: ', time() - parse_time)
    print('Executing...')

    svg_time = time()



    svg_document = svgwrite.Drawing(filename='output/map.svg', size=(str(8000)+'px', str(8000)+'px'))
    for way in ways:
        printed_nodes = []
        for nodeID in way:
            if nodeID in nodes:
                nodes[nodeID]['used'] = True
                printed_nodes.append((convertCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
                                                         float(nodes[nodeID]['lon']) - float(bounds['minlon']))))
        svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))

    # for nodeID in nodes:
    #     if nodes[nodeID]['used']:
    #         svg_document.add(svgwrite.shapes.Circle(
    #             center=(convertCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
    #                                         float(nodes[nodeID]['lon']) - float(bounds['minlon']))), r=1, fill='red'))
    # svg_document.save()

    # print('Svg write time: ', time() - svg_time)
    # print('Executing...')

    adj_list_time = time()

    adj_list = write_adj_list(nodes, ways)

    print('Adj list write time: ', time() - adj_list_time)
    print('Executing...')

    for nodeID in adj_list:
        if nodeID in nodes and nodes[nodeID]['used']:
            svg_document.add(svgwrite.shapes.Circle(
                center=(convertCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
                                           float(nodes[nodeID]['lon']) - float(bounds['minlon']))), r=1, fill='red'))

    svg_document.save()
    adj_matrix_time = time()

    fieldnames = list(adj_list.keys())

    # with open('output/adjacency_matrix.csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=[''] + fieldnames)
    #
    #     writer.writeheader()
    #     for node in fieldnames:
    #         written_row = {neighbour: (1 if neighbour in adj_list[node] else 0) for neighbour in fieldnames}
    #         written_row[''] = node
    #         writer.writerow(written_row)
    #
    # print('Adj matrix time: {} min'.format((time() - adj_matrix_time) / 60))


if __name__ == "__main__":
    start_time = time()
    print('Executing...')
    main()
    print('Finished!')
    print('Total time: ', time()-start_time)
