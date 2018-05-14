import csv
import svgwrite
import sys
from GraphOSM import GraphOSM
from GraphOSM import coordinate_input
from time import time


main_time = time()
print('Executing...')

if len(sys.argv) == 1:
    print('There are no input data!')
    sys.exit()
    pass
else:
    file = sys.argv[1]

city = GraphOSM()

city.parse(file)
city.generate_adjlist()

lat, lon = coordinate_input((city.bounds['minlat'], city.bounds['maxlat']),
                            (city.bounds['minlon'], city.bounds['maxlon']))

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