import random
from re import L
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import os
import cv2 as cv
import csv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
from glob import glob
import torch as th
import math
import matplotlib.pyplot as plt

def rotate_points(points, indices, should):
    indices = np.argmax(indices,1)
    indices[indices == 0] = 1000
    unique_indices = np.unique(indices)
    num_unique_indices = len(unique_indices)
    rotated_points = np.zeros_like(points)
    rotation_angles = []
    for i in unique_indices:
        idx = (indices == i)
        selected_points = points[idx]
        rotation_angle = 0 if i == 1 else (np.random.rand() * 360)
        if not should:
            rotation_angle = 0 
        # rotation_angle = 0 if i =0 else (np.random.randint(4) * 90)
        rotation_angle = np.deg2rad(rotation_angle)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)], # this is selected for return
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        rotated_selected_points = np.matmul(rotation_matrix, selected_points.T).T
        rotated_points[idx] = rotated_selected_points
        # rotation_matrix[0,1] = 1 if rotation_angle<np.pi else -1
        rotation_angles.extend(rotation_matrix[0:1].repeat(rotated_selected_points.shape[0], axis=0))
    return rotated_points, rotation_angles


def load_cube_data(
    batch_size,
    set_name,
    rotation 
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of cube...")
    deterministic = False if set_name == 'train' else True
    dataset = cube(set_name, rotation = rotation)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = False
        )
    else:
        loader = DataLoader(
            dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = False
        )
    while True:
        yield from loader

# roomshape  #center  #index  
class cube(Dataset):
    def __init__(self, set_name, rotation):
        super().__init__()
        max_num_points = 1000
        if set_name == "train":
            
            path = '../datasets/cube/training' # for cube dataset test
        else:
            path = '../datasets/cube/jsons_test'
        open('output.txt', 'w')
        print(path)
        self.set_name = set_name
        self.rotation = True # rotation
        self.puzzles = []
        self.rels = []
        volumes = {}
        pairss = {}
        files = glob(f'{path}/*')
        print(files)
        files = [x.split('/')[-1][:-4].split('_') for x in files]
        notused = set()
        #num_p_c = np.zeros(20)
        num_p_c = np.zeros(max_num_points)
        num_h_min = 12345678
        num_h_max = -1
        num_h_sum = []
        num_av = []
        min_num_av = 123456
        max_num_av = -1
        
        for file_name in tqdm(files, desc ='loading data files'):
            used = True
            image_size =[0,0]
            file_number = file_name[1][:-1]
            if file_number not in volumes:
                volumes[file_number] = []
                f = open(f'{path}/{file_name[0]}_{file_number}.json')
                json_data = json.load(f)
                f.close()
                pairs = []
                markers_by_slice_number = {}
                if (1 + int(list(json_data.keys())[-1])) <= 3:
                    continue
                cummulative_marker_counter = 0
                all_volume_markers = []
                num_p_c[1 + int(list(json_data.keys())[-1])] += 1
                num_av_t = 0
                num_av_c = 0
                #print(list(json_data.keys()))
                for slice_number_str in list(json_data.keys()):
                    slice_number = int(slice_number_str)
                    if slice_number == 0:
                        continue
                    
                    slice_data = json_data[str(slice_number)]
                    if(len(slice_data) < 3):
                        used = False
                        notused.add(int(file_number[:-1]))
                        volumes[file_number] = []
                        continue
                    img_size = json_data["0"]
                    #print(img_size)
                    image_size = img_size
                    wxx = 2 * img_size[0]/256.0
                    wyy = 2 * img_size[1]/256.0
                    #wzz = 2 * img_size[2]/256.0
                    markers_by_slice_number[slice_number] = []  
                    if (len(slice_data) > max_num_av):
                        max_num_av = len(slice_data)
                    if (len(slice_data) < min_num_av):
                        min_num_av = len(slice_data)
                    num_av_t += len(slice_data) 
                    num_av_c += 1 
                    for marker in slice_data:
                        all_volume_markers.append([marker[0], cummulative_marker_counter, slice_number])
                        markers_by_slice_number[slice_number].append([marker[0], cummulative_marker_counter, slice_number])
                        cummulative_marker_counter += 1
                    if used == True:
                        normalized_slice_coordinates = []
                        for marker in slice_data:
                            ax = 0 #random.uniform(-2, 2)*img_size[0]/256.0   ## change it if you want to add noise
                            ay = 0 #random.uniform(-2, 2)*img_size[1]/256.0  ## change it if you want to add noise
                            normalized_x_coordinate = (marker[0][0] + ax) / img_size[0]
                            normalized_y_coordinate = (marker[0][1] + ay) / img_size[1]
                            normalized_slice_coordinates.append([normalized_x_coordinate,normalized_y_coordinate])
                        cx = np.mean(np.array(normalized_slice_coordinates)[:,0])
                        cy =  np.mean(np.array(normalized_slice_coordinates)[:,1])
                        #cz = 0
                        volumes[file_number].append( {'normalized_slice_coordinates' : np.array(normalized_slice_coordinates) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': file_number})
                if num_av_c != 0:
                    num_av.append(num_av_t/num_av_c)        
                pairs = []
                for slice_i in markers_by_slice_number.keys():
                        slice_marker_data = markers_by_slice_number[slice_i]
                        for marker_data in slice_marker_data:
                            minimum_distance = 10000
                            pair_b = -1
                            marker_i = marker_data[0]
                            index = marker_data[1]
                            room_index = marker_data[2]
                            for nn in all_volume_markers:
                                point_b = nn[0]
                                #print(point , point_b, all_numbers)
                                index_b = nn[1]
                                room_index_b = nn[2]
                                if room_index == room_index_b:
                                    continue
                                if abs(math.dist(marker_i , point_b)) < minimum_distance and abs(math.dist(marker_i , point_b)) < 10*abs(math.sqrt(wxx**2 + wyy**2)):
                                    minimum_distance = abs(math.dist(marker_i , point_b))
                                    pair_b = index_b
                            if pair_b!=-1 :
                                pairs.append([index, pair_b])
                        pairss[file_number] = pairs

                        if image_size[0] < 50 or image_size[1] < 50:
                            pairss[file_number] = []
                            volumes[file_number] = []
                        """   
                        if len(all_numbers) > 100 or len(all_numbers) < 10 or len(pairs) > 100 :
                            pairss[file_number] = []
                            volumes[file_number] = []
                        """
        keyss = volumes.keys()
        self.puzzles1 = []
        self.rels = []
        with open('output.txt', 'a') as f:
            f.write(f"keyys: {keyss}\n\n")
            f.write(f"volumes: {volumes}\n\n")
            f.write(f"pairss: {pairss}\n\n")
            f.write(f"all_numbers: {all_volume_markers}\n\n")
        print(len(pairss["1"]))
        for key in keyss:
            if len(volumes[key]) > 1 and len(pairss[key]) >= 3:
                self.puzzles1.append(volumes[key])
                padding = np.zeros((1000-len(pairss[key]), 2))
                #padding = np.zeros((100, 2)) # did this since I have not figured out the matching loss yet
                rel = np.concatenate((np.array(pairss[key]), padding), 0)
                #rel = padding
                self.rels.append(rel)

        get_one_hot = lambda x, z: np.eye(z)[x]
        puzzles = []
        self_masks = []
        gen_masks = []
        for p in (self.puzzles1):
            puzzle = []
            corner_bounds = []
            num_points = 0
            for i, piece in enumerate(p):
                normalized_slice_coordinates = piece['normalized_slice_coordinates']
                center = np.ones_like(normalized_slice_coordinates) * piece['center']
                # Adding conditions
                num_piece_corners = len(normalized_slice_coordinates)
                piece_index = np.repeat(np.array([get_one_hot(len(puzzle)+1, 32)]), num_piece_corners, 0)
                corner_index = np.array([get_one_hot(x, 32) for x in range(num_piece_corners)])
                # Adding rotation
                if self.rotation:
                    normalized_slice_coordinates, angles = rotate_points(normalized_slice_coordinates, piece_index, True)
                # Src_key_padding_mask
                padding_mask = np.repeat(1, num_piece_corners)
                padding_mask = np.expand_dims(padding_mask, 1)
                # Generating corner bounds for attention masks
                connections = np.array([[i,(i+1)%num_piece_corners] for i in range(num_piece_corners)])
                connections += num_points
                corner_bounds.append([num_points, num_points+num_piece_corners])
                num_points += num_piece_corners
                piece = np.concatenate((center, angles, normalized_slice_coordinates, corner_index, piece_index, padding_mask, connections), 1)
                puzzle.append(piece)
            
            puzzle_layouts = np.concatenate(puzzle, 0)
            if len(puzzle_layouts) > max_num_points:
                assert False
            num_h_sum.append(len(puzzle_layouts))
            if num_h_min > len(puzzle_layouts):
                num_h_min = len(puzzle_layouts)
            if num_h_max < len(puzzle_layouts):
                num_h_max = len(puzzle_layouts)
            padding = np.zeros((max_num_points-len(puzzle_layouts), 73))
            gen_mask = np.ones((max_num_points, max_num_points))
            gen_mask[:len(puzzle_layouts), :len(puzzle_layouts)] = 0
            puzzle_layouts = np.concatenate((puzzle_layouts, padding), 0)
            self_mask = np.ones((max_num_points, max_num_points))
            for i in range(len(corner_bounds)):
                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[i][0]:corner_bounds[i][1]] = 0
            puzzles.append(puzzle_layouts)
            self_masks.append(self_mask)
            gen_masks.append(gen_mask)
        with open('output.txt', 'a') as f:
            f.write(f"puzzles: {puzzles}\n\n")
            f.write(f"normalized_slice_coordinates: {normalized_slice_coordinates}\n\n")
            f.write(f"self.puzzles1: {self.puzzles1}\n\n")

        
        self.max_num_points = max_num_points
        self.puzzles = puzzles
        self.self_masks = self_masks
        self.gen_masks = gen_masks
        self.num_coords = 4 # what is this for?
       
    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        arr = self.puzzles[idx][:, :self.num_coords]
        arr = np.transpose(arr, [1, 0]).astype(float)
        
        polys = self.puzzles[idx][:, self.num_coords:self.num_coords+2]
        cond = {
                'self_mask': self.self_masks[idx],
                'gen_mask': self.gen_masks[idx],
                # 'poly': self.puzzles[idx][:, self.num_coords:self.num_coords+2],
                'poly': polys,
                'corner_indices': self.puzzles[idx][:, self.num_coords+2:self.num_coords+34],
                'room_indices': self.puzzles[idx][:, self.num_coords+34:self.num_coords+66],
                'src_key_padding_mask': 1-self.puzzles[idx][:, self.num_coords+66],
                'connections': self.puzzles[idx][:, self.num_coords+67:self.num_coords+69],
                'rels': self.rels[idx],
                }
        
        return arr, cond

if __name__ == '__main__':
    dataset = cube('test')
