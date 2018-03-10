from lxml import etree
from time import time
import svgwrite
import csv
import sys


def convertCoordinate(coordinate):
    return coordinate * 1500


def parse(file):
    bounds = {}     # {'minlat': str, 'minlon': str, 'maxlat': str, 'maxlot': str}
    nodes = {}      # {'id': {'lat': str, 'lon': str, used: bool}}
    ways = []       # [[], ... ,[]]
    highways = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                'unclassified', 'residential', 'service', 'road']

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

    with open('adjacency_list.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for ID in adj_list:
            writer.writerow([ID] + adj_list[ID])


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

    svg_document = svgwrite.Drawing(filename='map.svg')
    for way in ways:
        printed_nodes = []
        for nodeID in way:
            if nodeID in nodes:
                nodes[nodeID]['used'] = True
                printed_nodes.append((convertCoordinate(float(nodes[nodeID]['lon']) - float(bounds['minlon'])),
                                      convertCoordinate(float(nodes[nodeID]['lat']) - float(bounds['minlat']))))

        svg_document.add(svgwrite.shapes.Polyline(printed_nodes, fill='none', stroke='black', stroke_width=0.5))

    # for nodeID in nodes:
    #     if nodes[nodeID]['used']:
    #         svg_document.add(svgwrite.shapes.Circle(
    #             center=(convertCoordinate(float(nodes[nodeID]['lon']) - float(bounds['minlon'])),
    #                     convertCoordinate(float(nodes[nodeID]['lat']) - float(bounds['minlat']))),
    #             r=0.2, fill='red'))

    svg_document.save()

    print('Svg write time: ', time() - svg_time)
    print('Executing...')

    adj_list_time = time()

    write_adj_list(ways)

    print('Adj list write time: ', time() - adj_list_time)
    print('Executing...')

    # fieldnames = list(adj_list.keys())
    #
    #
    # with open('adjacency_matrix2.csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=[''] + fieldnames)
    #     writer.writeheader()
    #     for node in fieldnames:
    #         written_row = {}
    #         written_row = {neighbour: 1 for neighbour in adj_list[node]}
    #         written_row[''] = node
    #         writer.writerow(written_row)


if __name__ == "__main__":
    start_time = time()
    print('Executing...')
    main()
    print('Finished!')
    print('Total time: ', time()-start_time)
