import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import os
import json
from torch_geometric.data import Dataset
import random
import math

def get_midpoints(cx, cy, w, h):
    left_mid = (cx - w / 2, cy)
    right_mid = (cx + w / 2, cy)
    top_mid = (cx, cy - h / 2)
    bottom_mid = (cx, cy + h / 2)
    top_left = (cx-w/2,cy-h/2)
    top_right=(cx+w/2,cy-h/2)
    bottom_left=(cx-w/2,cy+h/2)
    bottom_right=(cx+w/2,cy+h/2)
    return [left_mid, right_mid, top_mid, bottom_mid,top_left,top_right,bottom_left,bottom_right]

def compute_edge_mid_distance(center_a, center_b, box_a, box_b, lines):
    midpoints_a = get_midpoints(center_a[0], center_a[1], box_a[0], box_a[1])
    midpoints_b = get_midpoints(center_b[0], center_b[1], box_b[0], box_b[1])
    min_total_distance = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line['x1'],line['y1'],line['x2'],line['y2']
        min_dist_a_left = float('inf')
        min_dist_a_right = float('inf')
        for midpoint_a in midpoints_a:
            d1 = ((midpoint_a[0] - x1) ** 2 + (midpoint_a[1] - y1) ** 2) ** 0.5
            d2 = ((midpoint_a[0] - x2) ** 2 + (midpoint_a[1] - y2) ** 2) ** 0.5
            min_dist_a_left = min(min_dist_a_left, d1)
            min_dist_a_right = min(min_dist_a_right, d2)

        min_dist_b_left = float('inf')
        min_dist_b_right = float('inf')
        for midpoint_b in midpoints_b:
            d1 = ((midpoint_b[0] - x1) ** 2 + (midpoint_b[1] - y1) ** 2) ** 0.5
            d2 = ((midpoint_b[0] - x2) ** 2 + (midpoint_b[1] - y2) ** 2) ** 0.5
            min_dist_b_left = min(min_dist_b_left, d1)
            min_dist_b_right = min(min_dist_b_right, d2)

        total_distance_option_1 = min_dist_a_left + min_dist_b_right
        total_distance_option_2 = min_dist_a_right + min_dist_b_left
        min_total_distance = min(min_total_distance, total_distance_option_1, total_distance_option_2)
    return min_total_distance


def compute_edge_box_distance(center_a, center_b, box_a_info, box_b_info, lines):
    def point_to_box_distance(px, py, box):

        x_min, y_min, x_max, y_max = box
        if px < x_min:
            dist_x = x_min - px
        elif px > x_max:
            dist_x = px - x_max
        else:
            dist_x = 0
        if py < y_min:
            dist_y = y_min - py
        elif py > y_max:
            dist_y = py - y_max
        else:
            dist_y = 0

        return math.sqrt(dist_x ** 2 + dist_y ** 2)

    def get_box(cx, cy, w, h):
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]
    box_a=get_box(center_a[0], center_a[1], box_a_info[0], box_a_info[1])
    box_b=get_box(center_b[0], center_b[1], box_b_info[1], box_b_info[1])
    min_total_distance = float('inf')

    for line in lines:
        x1, y1, x2, y2 = line['x1'],line['y1'],line['x2'],line['y2']
        dist_i1 = point_to_box_distance(x1, y1, box_a)
        dist_j1 = point_to_box_distance(x2, y2, box_b)
        dist1 = dist_i1 + dist_j1
        dist_i2 = point_to_box_distance(x2, y2, box_a)
        dist_j2 = point_to_box_distance(x1, y1, box_b)
        dist2 = dist_i2 + dist_j2
        min_total_distance = min(min_total_distance,dist1, dist2)
    return min_total_distance

def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    objects = data['objects']
    lines = data['lines']
    connections = data['connections']

    node_features = []
    object_ids = list(objects.keys())
    for oid in object_ids:
        obj = objects[oid]
        node_features.append([obj['x'], obj['y'], obj['w'], obj['h'], obj['class_id']])

    edge_index = []
    edge_attr = []
    edge_label = []
    for i, id_a in enumerate(object_ids):
        for j, id_b in enumerate(object_ids):
            if i == j:
                continue

            center_a = (objects[id_a]['x'], objects[id_a]['y'])
            center_b = (objects[id_b]['x'], objects[id_b]['y'])
            box_a = (objects[id_a]['w'], objects[id_a]['h'])
            box_b = (objects[id_b]['w'], objects[id_b]['h'])

            #distance = compute_edge_box_distance(center_a, center_b, box_a, box_b, lines) #Box
            distance = compute_edge_mid_distance(center_a, center_b, box_a, box_b, lines)  #Mid
            label = 0
            for conn in connections:
                if (conn['id_a'] == int(id_a) and conn['id_b'] == int(id_b)) or \
                        (conn['id_b'] == int(id_a) and conn['id_a'] == int(id_b)):
                    label = 1
                    break

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(distance)
            edge_attr.append(distance)

            edge_label.append(label)
            edge_label.append(label)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    edge_label = torch.tensor(edge_label, dtype=torch.float).view(-1, 1)

    graph_data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr,
        edge_label=edge_label)
    return graph_data

def load_dataset(json_dir,random_data,percentage=25):
    graph_list = []
    file_list = []
    if random_data:
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                json_file_path = os.path.join(json_dir, filename)
                file_list.append(json_file_path)
        num_files_to_select = int(len(file_list) * (percentage / 100))
        selected_files = random.sample(file_list, num_files_to_select)
        for file in selected_files:
            graph_data = load_json_data(file)
            graph_list.append(graph_data)
        return graph_list
    else:
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                json_file = os.path.join(json_dir, filename)
                graph_data = load_json_data(json_file)
                graph_list.append(graph_data)
        return graph_list

class MultiGraphDataset(Dataset):
    def __init__(self, root_dir,random_data):
        super(MultiGraphDataset, self).__init__()
        self.data_list = load_dataset(root_dir,random_data)
    def len(self):
        return len(self.data_list)
    def get(self, idx):
        return self.data_list[idx]