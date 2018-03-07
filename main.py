from lxml import etree
from time import time
import svgwrite


def convertCoordinate(coordinate):
    return coordinate * 1500


def main():
    bounds = {}
    nodes = {}  # lat, lot, used
    ways = []  #

    for _, element in etree.iterparse('../../osmosis/krasnodar.osm', tag=['bounds', 'node', 'way']):
        if element.tag == 'bounds':
            bounds['minlat'] = element.get('minlat')
            bounds['minlon'] = element.get('minlon')
            bounds['maxlat'] = element.get('maxlat')
            bounds['maxlot'] = element.get('maxlon')
        elif element.tag == 'node':
            nodes[element.get('id')] = [element.get('lat'), element.get('lon'), False]
        elif element.tag == 'way':
            rfnodes = []
            for child in element.iter('nd', 'tag'):
                if child.tag == 'nd':
                    rfnodes.append(child.get('ref'))
                elif child.tag == 'tag' and child.get('k') == 'highway':
                    ways.append(rfnodes)
        element.clear()

    svg_document = svgwrite.Drawing(filename='map.svg')

    for way in ways:
        nodes_print = []
        for nodeID in way:
            if nodeID in nodes:
                nodes[nodeID][2] = True
                nodes_print.append((convertCoordinate(float(nodes[nodeID][1]) - float(bounds['minlon'])),
                              convertCoordinate(float(nodes[nodeID][0]) - float(bounds['minlat']))))
        svg_document.add(svgwrite.shapes.Polyline(nodes_print, fill='none', stroke='black'))
    svg_document.save()


if __name__ == "__main__":
    start_time = time()
    main()
    print("Total time: ", time()-start_time)
