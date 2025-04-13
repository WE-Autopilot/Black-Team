import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

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
plt.show()

for coord in coords:
    plt.imshow(image)

    a = np.random.uniform(-1, 1, (10, 2))
    a = a / np.linalg.norm(a, axis=1).max() * 1.5 / resolution
    b = a + coord

    plt.scatter(*coords.T)
    plt.scatter(*b.T)
    plt.scatter(*coord)
    plt.show()
