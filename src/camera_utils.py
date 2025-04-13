from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def crop(image, scale, rotation, dimensions, location):
  out_width, out_height = dimensions.astype(int)
  
  width, height = image.size

  left = (location[0]*scale) - out_width / 2
  right = (location[0]*scale) + out_width / 2
  upper = (location[1]*scale) + out_height / 2
  lower = (location[1]*scale) - out_height / 2
  print(left,lower,right,upper)
  plt.imshow(image)
  plt.show()
  cropped_image = image.rotate(np.degrees(rotation), resample=Image.BILINEAR, center=(location[0], location[1]))
  plt.imshow(cropped_image)
  plt.show()
  cropped_image = cropped_image.resize((np.array(image.size) * scale).astype(int), Image.Resampling.BILINEAR)

  cropped_image = cropped_image.crop((left,lower,right,upper))
  

  return cropped_image
image = Image.open("maps/map0.png")

cropped_image = crop(image, 2, 0.2*np.pi, np.array([256, 256]), np.array([466, 469]))
cropped_image.show()
print(cropped_image.size)
print(image.size)
