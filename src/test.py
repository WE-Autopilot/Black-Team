import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml


def draw_loop_connections(loop, m):
    n = len(loop)
    for i in range(n):
        j = (i + m) % n
        plt.plot([loop[i, 0], loop[j, 0]], [loop[i, 1], loop[j, 1]], 'b-')


def gen_perp_points(curr_wp, next_wp, num_points=10, min_dis=-1.5, max_dis=1.5):
    vector = next_wp - curr_wp
    perp = np.array([-vector[1], vector[0]])
    unit_perp = perp / np.linalg.norm(perp)
    distances = np.random.uniform(min_dis, max_dis, num_points)
    points = curr_wp + unit_perp * distances[:, np.newaxis]
    return points


index = 0
map_path = f"maps/map{index}"
#map_path = "../weap_maps/track1"
image = Image.open(f'{map_path}.png')
with open(f'{map_path}.yaml', 'r') as file:
    config = yaml.safe_load(file)
waypoints = np.loadtxt(f'maps/map{index}.csv', delimiter=',', skiprows=0)[:, :2]

origin = np.array(config["origin"])[:2]
resolution = config["resolution"]

spawn = -origin / resolution
spawn[1] = image.height - spawn[1]

coords = (waypoints - origin) / config["resolution"]
coords[:, 1] = image.height - coords[:, 1]

plt.imshow(image)
plt.scatter(*coords.T)
plt.scatter(*spawn)
draw_loop_connections(coords, 6)
plt.show()

plt.cla()
path_vecs = coords - np.roll(coords, 1, axis=0)
angles = np.arctan2(*path_vecs.T[::-1])
path_angles = angles - np.roll(angles, 1, axis=0)
plt.plot(path_angles)
plt.show()

for i in range(len(coords)):
    coord = coords[i]
    next_coord = coords[(i + 1) % len(coords)]
    plt.imshow(image)

    points = gen_perp_points(coord, next_coord, 10, -1.5 / 0.0625, 1.5 / 0.0625)

    plt.scatter(*coords.T)
    plt.scatter(*points.T)
    plt.scatter(*coord)
    plt.show()
