import os
import shutil
import json

# dataset_path_target = '/home/guests/hiran_sarkar/FullCode/datasets/DyNeRF_blender_data_ball_roll_multicam_interp/train/'
# dataset_path_source = '/home/guests/hiran_sarkar/FullCode/datasets/DyNeRF_blender_data_ball_roll_multicam_interp/all/'

# for k, i in enumerate(os.listdir(dataset_path_source)):
#     for n, j in enumerate(os.listdir(dataset_path_source + str(i) + '/train/')):
#         if int(j[2:].split('.')[0])%2 ==0:
#             # print(str(int(j[2:].split('.')[0])))
#             shutil.copy(dataset_path_source + str(i) + '/' + 'train/' + 'r_' + str(int(j[2:].split('.')[0])) + '.png', dataset_path_target + 'r_' + str(85*k+int(j[2:].split('.')[0])) + '.png')


dataset_json_target = '/home/guests/hiran_sarkar/FullCode/datasets/DyNeRF_blender_data_ball_roll_multicam_interp/transforms_train.json'
dataset_json_source = '/home/guests/hiran_sarkar/FullCode/datasets/DyNeRF_blender_data_ball_roll_multicam_interp/all/'

new_transform = {"camera_angle_x": 0.6911112070083618, "frames":[]}

for k, i in enumerate(os.listdir(dataset_json_source)):
    with open(dataset_json_source + str(i) + '/transforms_train.json', 'r') as f:
        a = json.load(f)
    for n, j in enumerate(a['frames']):
        if int(j['file_path'][10:])%2 == 0:
            j['file_path'] = './train/r_' + str(85*k + int(j['file_path'][10:]))
            new_transform["frames"].append(j)

with open(dataset_json_target, 'w') as json_file:
    json.dump(new_transform, json_file)

