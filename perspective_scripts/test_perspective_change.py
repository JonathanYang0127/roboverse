import roboverse
import numpy as np
from PIL import Image
from perspectives import perspectives
from roboverse.policies import policies


camera_params = perspectives[0]
#camera_params['camera_roll'] += 
#camera_params['camera_pitch'] += np.random.uniform(-20, 20) 
#camera_params['camera_yaw'] += np.random.uniform(-20, 20)
#camera_params['object_names'] = perspectives[1]['object_names']
#camera_params['object_scales'] = perspectives[1]['object_scales']
#camera_params['target_object'] = perspectives[1]['target_object']

env = roboverse.make('Widow250PickPlace-v0', gui=False, 
        observation_img_dim=128, **camera_params)
env.reset()
policy_class = policies['pickplace']
policy = policy_class(env)
policy.reset()

images = []
for _ in range(25):
    action, agent_info = policy.get_action()
    env_action_dim = env.action_space.shape[0]
    if env_action_dim - action.shape[0] == 1:
       action = np.append(action, 0)
    action = np.clip(action, -1, 1)
    print(action[6] > 0.5)
    env.step(action)


    obs = env.get_observation()
    img = obs['image'].reshape(3, 128, 128).transpose(1, 2, 0)
    img = (img * 255.0).astype(np.uint8)
    images.append(Image.fromarray(img))

images[0].save("out.gif", save_all=True, append_images=images[1:],
        duration=100, loop=0)


