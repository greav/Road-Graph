from lxml import etree
from time import time
import svgwrite
import csv
import sys
import math

def convertCoordinates(lat, lon):
    radius = 1
    multiplier = 1000000

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


def write_adj_list(ways):
    adj_list = {}
    for way in ways:
        if way[0] in adj_list:
            adj_list[way[0]].append(way[1])
        else:
            adj_list[way[0]] = [way[1]]
        i = 1
        while i < len(way) - 1:
            if way[i] in adj_list:
                adj_list[way[i]].extend([way[i-1], way[i+1]])
            else:
                adj_list[way[i]] = [way[i-1], way[i+1]]
            i = i + 1
        if way[i] in adj_list:
            adj_list[way[i]].append(way[i-1])
        else:
            adj_list[way[i]] = [way[i-1]]

    with open('adjacency_list_service.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for ID in adj_list:
            writer.writerow([ID] + adj_list[ID])

    return adj_list


def main():

    if len(sys.argv) == 1:
        file = '/home/victor/osmosis/krasnodar.osm'
    else:
        file = sys.argv[1]

    parse_time = time()

    bounds, nodes, ways = parse(file)

    print('Parse time: ', time() - parse_time)
    print('Executing...')

    svg_time = time()

    svg_document = svgwrite.Drawing(filename='map_service.svg', size=(str(8000)+'px', str(8000)+'px'))
    for way in ways:
        printed_nodes = []
        for nodeID in way:
            if nodeID in nodes:
                nodes[nodeID]['used'] = True
                printed_nodes.append((convertCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
                                                         float(nodes[nodeID]['lon']) - float(bounds['minlon']))))
                                    
        svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))

    for nodeID in nodes:
        if nodes[nodeID]['used']:
            svg_document.add(svgwrite.shapes.Circle(
                center=(convertCoordinates(float(nodes[nodeID]['lat']) - float(bounds['minlat']),
                                            float(nodes[nodeID]['lon']) - float(bounds['minlon']))),
                r=1, fill='red'))
    svg_document.save()

    print('Svg write time: ', time() - svg_time)
    print('Executing...')

    adj_list_time = time()

    adj_list = write_adj_list(ways)

    print('Adj list write time: ', time() - adj_list_time)
    print('Executing...')

    # adj_matrix_time = time()
    #
    # fieldnames = list(adj_list.keys())
    #
    #
    # with open('test.csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=[''] + fieldnames)
    #
    #     writer.writeheader()
    #     for node in fieldnames:
    #         written_row = {neighbour: 1 for neighbour in adj_list[node]}
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
