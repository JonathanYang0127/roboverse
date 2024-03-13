import numpy as np
from roboverse.assets.shapenet_object_lists import PICK_PLACE_TRAIN_OBJECTS

perspectives = [
    {
        'camera_target_pos': (0.6, 0.2, -0.28),
        'camera_distance': 0.29,
        'camera_roll': 0.0,
        'camera_pitch': -40,
        'camera_yaw': 180,

        'object_names': ('shed',),
        'object_scales': (0.7,),
        'target_object': 'shed',
    },
    {
        'camera_target_pos': (0.7, 0.2, -0.28),
        'camera_distance': 0.4,
        'camera_roll': 0.0,
        'camera_pitch': -20,
        'camera_yaw': 145,

        'object_names': (PICK_PLACE_TRAIN_OBJECTS[2],),
        'object_scales': (0.5,),
        'target_object': PICK_PLACE_TRAIN_OBJECTS[2],
    },
    {
        'camera_target_pos': (0.6, 0.2, -0.2),
        'camera_distance': 0.4,
        'camera_roll': 0.0,
        'camera_pitch': -60,
        'camera_yaw': 200,

        'object_names': (PICK_PLACE_TRAIN_OBJECTS[4],),
        'object_scales': (0.7,),
        'target_object': PICK_PLACE_TRAIN_OBJECTS[4],
    },
    {
        'camera_target_pos': (0.6, 0.2, -0.3),
        'camera_distance': 0.35,
        'camera_roll': 0.0,
        'camera_pitch': -30,
        'camera_yaw': 210,

        'object_names': (PICK_PLACE_TRAIN_OBJECTS[7],),
        'object_scales': (0.7,),
        'target_object': PICK_PLACE_TRAIN_OBJECTS[7],
    },
    {
        'camera_target_pos': (0.6, 0.2, -0.35),
        'camera_distance': 0.35,
        'camera_roll': 0.0,
        'camera_pitch': -15,
        'camera_yaw': 180,

        'object_names': (PICK_PLACE_TRAIN_OBJECTS[9],),
        'object_scales': (0.7,),
        'target_object': PICK_PLACE_TRAIN_OBJECTS[9],
    },
    {
        'camera_target_pos': (0.6, 0.2, -0.24),
        'camera_distance': 0.35,
        'camera_roll': 0.0,
        'camera_pitch': -35,
        'camera_yaw': 170,

        'object_names': (PICK_PLACE_TRAIN_OBJECTS[10],),
        'object_scales': (0.7,),
        'target_object': PICK_PLACE_TRAIN_OBJECTS[10],
    }

]

