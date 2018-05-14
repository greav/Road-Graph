import csv
import random
import svgwrite
import sys
from GraphOSM import GraphOSM
from GraphOSM import coordinate_input
from time import time


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


main_time = time()
print('Executing...')

if len(sys.argv) == 1:
    print('There are no input data!')
    # sys.exit()
else:
    file = sys.argv[1]

file = 'krasnodar.osm'

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


all_dijkstra_distances = [x for x in all_dijkstra_distances if x != float('inf')]
all_levit_distances= [x for x in all_levit_distances if x != float('inf')]
all_astar_distances = [x for x in all_astar_distances if x != float('inf')]

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
