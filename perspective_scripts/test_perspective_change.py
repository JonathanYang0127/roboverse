import roboverse
import numpy as np
from PIL import Image
from perspectives import perspectives

camera_params = perspectives[1]
env = roboverse.make('Widow250GraspEasy-v0', gui=False, **camera_params)
env.reset()
images = []
for _ in range(25):
    env.step(env.action_space.sample())
    obs = env.get_observation()
    img = obs['image'].reshape(3, 48, 48).transpose(1, 2, 0)
    img = (img * 255.0).astype(np.uint8)
    images.append(Image.fromarray(img))

images[0].save("out.gif", save_all=True, append_images=images[1:],
        duration=100, loop=0)


