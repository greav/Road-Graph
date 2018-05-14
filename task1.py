import sys
from GraphOSM import GraphOSM
from time import time

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

city.adjlist_to_csv('temp.csv')
# city.adjmatrix_to_csv()

city.create_svgmap('krasnodar_map.svg')
city.svg_document.save()

print('Finished!')
print('Total time: ', time() - main_time)