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
import h5py
import multiprocessing
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


def load_voronoi_data(
    batch_size,
    set_name,
    rotation 
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    #print(f"loading {set_name} of voronoi...")
    deterministic = False if set_name == 'train' else True
    dataset = voronoi(set_name, rotation = rotation)
    num_workers = round(os.cpu_count()/2)  # Get the number of available CPUs
    if deterministic:
        loader = DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = num_workers, 
            drop_last = True
        )
    else:
        loader = DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = True,  
            drop_last = True
        )
    while True:
        yield from loader

# roomshape  #center  #index  
class voronoi(Dataset):
    def __init__(self, set_name, rotation = True, load = False):
        super().__init__()
        
        if set_name == "train":
            path = '../datasets/voronoi/jsons'
            #path = '../datasets/voronoi/mini_jsons'
        else:
            path = '../datasets/voronoi/jsons_test'
        #print(path)
        self.max_num_points = 1000
        self.set_name = set_name
        self.rotation = True # rotation
        self.puzzles = []
        self.rels = []
        if load:
            self.load_dict()
        else:
            """# Define a worker function to process a single file    
            # Get the list of files to process
            files = glob(f'{path}/*.json')
            
            # Define the number of worker processes
            num_workers = multiprocessing.cpu_count()
            
            # Create a pool of worker processes
            pool = multiprocessing.Pool(processes=num_workers)
            
            # Use the pool to process the files in parallel
            pool.map(process_file, files)
            
            # Close the pool to release resources
            pool.close()
            pool.join()"""
            
            self.process_data(path)

        
    def process_data(self, path):    
        houses = {}
        pairss = {}
        files = glob(f'{path}/*.json')
        self.file_count = len(files)
        #print(files)
        files = [x.split('/')[-1][:-4].split('_') for x in files]
        notused = set()
        #num_p_c = np.zeros(20)
        num_p_c = np.zeros(self.max_num_points)
        num_h_min = 12345678
        num_h_max = -1
        num_h_sum = []
        num_av = []
        min_num_av = 123456
        max_num_av = -1
        
        for name in tqdm(files, desc ='loading data files'):
            used = True
            image_size =[0,0]
            name[1] = name[1][:-1]
            if name[1] not in houses:
                houses[name[1]] = []
                with open(f'{path}/{name[0]}_{name[1]}.json') as f:
                    cnt = json.load(f)
                pairs = []
                numbers = {}
                if (1 + int(list(cnt.keys())[-1])) <= 3:
                    continue
                hss = 0
                all_numbers = []
                num_p_c[1 + int(list(cnt.keys())[-1])] += 1
                num_av_t = 0
                num_av_c = 0
                for i in range(1, 1 + int(list(cnt.keys())[-1])):
                    contours = cnt[str(i)]
                    if( len(contours)<3):
                        used = False
                        notused.add(int(name[1][:-1]))
                        houses[name[1]] = []
                        continue
                    img_size = cnt["0"]
                    image_size = img_size
                    wxx = 2 * img_size[0]/256.0
                    wyy = 2 * img_size[1]/256.0
                    numbers[i] = []  
                    if (len(contours) > max_num_av):
                        max_num_av = len(contours)
                    if (len(contours) < min_num_av):
                        min_num_av = len(contours)
                    num_av_t += len(contours) 
                    num_av_c += 1 
                    for cnc in contours:
                            all_numbers.append([cnc[0], hss, i])
                            numbers[i].append([cnc[0], hss, i])
                            hss += 1
                    if used == True:
                        poly = []
                        for cntt in contours:
                            ax = 0 #random.uniform(-2, 2)*img_size[0]/256.0   ## change it if you want to add noise
                            ay = 0 #random.uniform(-2, 2)*img_size[1]/256.0  ## change it if you want to add noise
                            a = (cntt[0][0] +ax) / img_size[0]
                            b = (cntt[0][1]+ay) / img_size[1]
                            poly.append([a,b])
                        cx = np.mean(np.array(poly)[:,0])
                        cy =  np.mean(np.array(poly)[:,1])
                        houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1]})
                del cnt
                del contours
                
                if num_av_c != 0:
                    num_av.append(num_av_t/num_av_c)        
                pairs = []
                for tk in numbers.keys():
                        number = numbers[tk]
                        for a in number:
                            min_diss =10000
                            pair_b = -1
                            point = a[0]
                            index = a[1]
                            room_index = a[2]
                            for  nn in all_numbers:
                                point_b = nn[0]
                                #print(point , point_b, all_numbers)
                                index_b = nn[1]
                                room_index_b = nn[2]
                                if room_index == room_index_b:
                                    continue
                                if abs(math.dist(point , point_b)) < min_diss and abs(math.dist(point , point_b)) < 10*abs(math.sqrt(wxx**2 + wyy**2)):
                                    min_diss = abs(math.dist(point , point_b))
                                    pair_b = index_b
                            if pair_b!=-1 :
                                pairs.append([index, pair_b])
                        pairss[name[1]] = pairs
                        pairss[name[1]] = pairs
                        if image_size[0] < 50 or image_size[1] < 50:
                            pairss[name[1]] = []
                            houses[name[1]] = []
                        if len(all_numbers) > 100 or len(all_numbers) < 10 or len(pairs) > 100 :
                            pairss[name[1]] = []
                            houses[name[1]] = []
        del numbers
        del all_numbers
        del number
        print("checkpoint 1")
        keyss = houses.keys()
        self.puzzles1 = []
        self.rels = []
        for ke in keyss:
            if len(houses[ke]) > 1 and len(pairss[ke]) >= 3:
                self.puzzles1.append(houses[ke])
                padding = np.zeros((100-len(pairss[ke]), 2))
                rel = np.concatenate((np.array(pairss[ke]), padding), 0)
                self.rels.append(rel)
        del pairss
        del houses
        del keyss
        del rel
        get_one_hot = lambda x, z: np.eye(z)[x]
        puzzles = []
        self_masks = []
        gen_masks = []
        print("checkpoint 2")
        for p in (self.puzzles1):
            puzzle = []
            corner_bounds = []
            num_points = 0
            for i, piece in enumerate(p):
                poly = piece['poly']
                center = np.ones_like(poly) * piece['center']
                # Adding conditions
                num_piece_corners = len(poly)
                piece_index = np.repeat(np.array([get_one_hot(len(puzzle)+1, 32)]), num_piece_corners, 0)
                corner_index = np.array([get_one_hot(x, 32) for x in range(num_piece_corners)])
                # Adding rotation
                if self.rotation:
                    poly, angles = rotate_points(poly, piece_index, True)
                # Src_key_padding_mask
                padding_mask = np.repeat(1, num_piece_corners)
                padding_mask = np.expand_dims(padding_mask, 1)
                # Generating corner bounds for attention masks
                connections = np.array([[i,(i+1)%num_piece_corners] for i in range(num_piece_corners)])
                connections += num_points
                corner_bounds.append([num_points, num_points+num_piece_corners])
                num_points += num_piece_corners
                piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections), 1)
                puzzle.append(piece)
            
            puzzle_layouts = np.concatenate(puzzle, 0)
            if len(puzzle_layouts) > self.max_num_points:
                assert False
            num_h_sum.append(len(puzzle_layouts))
            if num_h_min > len(puzzle_layouts):
                num_h_min = len(puzzle_layouts)
            if num_h_max < len(puzzle_layouts):
                num_h_max = len(puzzle_layouts)
            padding = np.zeros((self.max_num_points-len(puzzle_layouts), 73))
            gen_mask = np.ones((self.max_num_points, self.max_num_points))
            gen_mask[:len(puzzle_layouts), :len(puzzle_layouts)] = 0
            puzzle_layouts = np.concatenate((puzzle_layouts, padding), 0)
            self_mask = np.ones((self.max_num_points, self.max_num_points))
            for i in range(len(corner_bounds)):
                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[i][0]:corner_bounds[i][1]] = 0
            puzzles.append(puzzle_layouts)
            self_masks.append(self_mask)
            gen_masks.append(gen_mask)
            #print(f"Puzzles: {puzzles}\n\n")
            #print(f"Houses: {houses}\n\n")
            #print(f"Pairss: {pairss}\n\n")
            #print(f"Poly: {poly}\n\n")
            #print(f"Self.puzzles1: {self.puzzles1}\n\n")
            #print(f"All numbers: {all_numbers}\n\n")
        """with open('dataloading_output.txt', 'w') as f:
            f.write(f"Puzzles: {puzzles}\n\n")
            f.write(f"Houses: {houses}\n\n")
            f.write(f"Pairss: {pairss}\n\n")
            f.write(f"Poly: {poly}\n\n")
            f.write(f"Self.puzzles1: {self.puzzles1}\n\n")
            f.write(f"All numbers: {all_numbers}\n\n")"""
        del self.puzzles1
        del poly
        
        print("checkpoint 3")
        self.max_num_points = self.max_num_points
        self.puzzles = puzzles
        del puzzles
        self.self_masks = self_masks
        del self_masks
        self.gen_masks = gen_masks
        del gen_masks
        self.num_coords = 4 # what is this for?
        return None
        
    def save_dict(self):
        with h5py.File(f'data_dict_{self.file_count}.h5', 'w') as f:
            f.create_dataset('puzzles', data=self.puzzles)
            f.create_dataset('self_masks', data=self.self_masks)
            f.create_dataset('gen_masks', data=self.gen_masks)
            f.create_dataset('num_coords', data=self.num_coords)
            f.create_dataset('rels', data=self.rels)

    def load_dict(self):
        with h5py.File('data_dict.h5', 'r') as f:
            self.puzzles = f['puzzles'][:]
            self.self_masks = f['self_masks'][:]
            self.gen_masks = f['gen_masks'][:]
            self.num_coords = f['num_coords'][:]
            self.rels = f['rels'][:]
            
    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        arr = self.puzzles[idx][:, :self.num_coords]
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
        arr = np.transpose(arr, [1, 0])
        return arr.astype(float), cond

if __name__ == '__main__':
    dataset = voronoi('test')
