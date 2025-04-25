import os
import json
import torch
from torch_geometric.data import Data
import numpy as np
from GNN_structure import GNNModel_1layer,GNNModel_4layer,GNNModel_2layer,GNNModel_3layer
from data_loader import compute_edge_mid_distance,compute_edge_box_distance
import time
import random
import re
import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance
import cv2

def draw_objects_and_lines(json_file_path, image_file_path, output_image_path,draw_objects=None,draw_lines=None):
    """
    读取 JSON 文件中的 objects 和 lines 信息，并在图片上绘制。
    :param json_file_path: 输入 JSON 文件路径
    :param image_file_path: 输入图片路径
    :param output_image_path: 输出绘制结果的图片路径
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    image = cv2.imread(image_file_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_file_path}")
    if draw_objects:
        for obj_id, obj in data["objects"].items():
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.putText(image, f"Obj {obj_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if draw_lines:
        for line in data["lines"]:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            mid_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            #cv2.putText(image, f"Line {line['line_id']}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_image_path, image)
    print(f"Output saved to {output_image_path}")

def get_predict_score(edge_index,probabilities,x,y):
    node_a = x
    node_b = y
    for idx, (start, end) in enumerate(edge_index.t()):
        if (start == node_a and end == node_b) or (start == node_b and end == node_a):
            print(f"Prediction score for edge ({node_a}, {node_b}): {probabilities[idx].item()}")
            break

def predict_connections(model, json_file, threshold=0.9, device='cuda',distance_type="mid"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    objects = data['objects']
    lines = data['lines']
    connection_gt= data['connections']

    object_ids = list(objects.keys())
    node_features = []
    edge_index = []
    edge_attr = []

    for oid in object_ids:
        obj = objects[oid]
        node_features.append([obj['x'], obj['y'], obj['w'], obj['h'], obj['class_id']])

    for i, id_a in enumerate(object_ids):
        for j, id_b in enumerate(object_ids):
            if i == j:
                continue  # 跳过自连接

            center_a = (objects[id_a]['x'], objects[id_a]['y'])
            center_b = (objects[id_b]['x'], objects[id_b]['y'])
            box_a = (objects[id_a]['w'], objects[id_a]['h'])
            box_b = (objects[id_b]['w'], objects[id_b]['h'])

            if distance_type=="mid":
                distance = compute_edge_mid_distance(center_a, center_b, box_a, box_b, lines)
            else:
                distance = compute_edge_box_distance(center_a, center_b, box_a, box_b, lines)

            edge_index.append([i, j])
            edge_attr.append(distance)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    node_features = torch.tensor(node_features, dtype=torch.float)

    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    model.eval()
    model=model.to(device)
    graph_data = graph_data.to(device)
    with torch.no_grad():
        predictions = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        probabilities = torch.sigmoid(predictions)

    connections = set()
    for idx, prob in enumerate(probabilities):
        if prob > threshold:
            start, end = edge_index[:, idx]
            id_a, id_b = object_ids[start], object_ids[end]
            connection = tuple(sorted((id_a, id_b)))
            connections.add(connection)

    #  查看 x 和 y 之间的连接分数预测
    #get_predict_score(edge_index,probabilities,x,y)

    connections = sorted(list(connections), key=lambda x: (int(x[0]), int(x[1])))

    connections = list(connections)

    return connections,connection_gt

def calculate_nca(predicted_set, gt_set):
    G_pred = nx.Graph()
    G_gt = nx.Graph()
    G_pred.add_edges_from(predicted_set)
    G_gt.add_edges_from(gt_set)
    all_nodes = set(G_pred.nodes()).union(set(G_gt.nodes()))
    correct_connections = 0
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            if i not in G_pred or j not in G_pred:
                pred_connected = False  # 如果节点不在 G_pred 中，认为没有路径
            else:
                pred_connected = nx.has_path(G_pred, i, j)
            # 检查节点是否在图 G_gt 中
            if i not in G_gt or j not in G_gt:
                gt_connected = False  # 如果节点不在 G_gt 中，认为没有路径
            else:
                gt_connected = nx.has_path(G_gt, i, j)
            # 如果预测和真实连接状态一致，则计数
            if pred_connected == gt_connected:
                correct_connections += 1
    total_pairs = len(all_nodes) * (len(all_nodes) - 1)
    nca = correct_connections / total_pairs if total_pairs > 0 else 0.0
    return nca

def calculate_normalized_ged(predicted_set, gt_set):
    G_pred = nx.Graph()
    G_gt = nx.Graph()
    G_pred.add_edges_from(predicted_set)
    G_gt.add_edges_from(gt_set)
    ged = graph_edit_distance(G_pred, G_gt)
    max_ged = len(G_pred.nodes()) + len(G_gt.nodes()) + len(G_pred.edges()) + len(G_gt.edges())
    normalized_ged = ged / max_ged if max_ged > 0 else 0.0
    return normalized_ged

def connection_eval(connections,gt,eval_extra=False):
    predicted_set = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in connections)
    gt_set = set((min(entry['id_a'], entry['id_b']), max(entry['id_a'], entry['id_b'])) for entry in gt)

    NCA = calculate_nca(predicted_set, gt_set)
    normalized_ged = calculate_normalized_ged(predicted_set, gt_set)

    true_positives = predicted_set.intersection(gt_set)
    false_positives = predicted_set - gt_set
    false_negatives = gt_set - predicted_set
    accuracy = len(true_positives) / len(gt_set) if len(gt_set) > 0 else 0
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(
        false_positives) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(
        false_negatives) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1, true_positives, false_positives, false_negatives,normalized_ged,NCA


def  calculate_total_metrics(all_metrics):
    total_true_positives = sum([len(metric[4]) for metric in all_metrics])  # 累加每个图的正确预测连接数（长度）
    total_false_positives = sum([len(metric[5]) for metric in all_metrics])  # 累加每个图的误检连接数（长度）
    total_false_negatives = sum([len(metric[6]) for metric in all_metrics])  # 累加每个图的漏检连接数（长度）

    total_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    total_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_accuracy = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

    total_GED = sum([metric[7] for metric in all_metrics])/len(all_metrics)
    total_NCA = sum([metric[8] for metric in all_metrics])/len(all_metrics)

    return total_accuracy, total_precision, total_recall, total_f1,total_GED,total_NCA

def save_metrics_to_file(metrics, file_path):
    formatted_metrics = "，".join(f"{key}：{value:.4f}" for key, value in metrics.items())
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(formatted_metrics)
    print(f"The metrics save to file: {file_path}")

if __name__ == "__main__":

    info_dir = ""  #json dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir=""   #image dir

    percentage=10
    RANDOM=False
    SELECT=False
    # start_frame=
    # end_frame=

    model_name="connection_pred_epoch15_layer2_mid" #model name
    model_path = "" + model_name+".pth"
    output_dir = ""
    output_image=False
    distance_type="box"
    eval_extra=True  # calculate GED and NCA

    output_folder=output_dir+model_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time=time.time()
    model = GNNModel_2layer() #GNNModel_1layer, GNNModel_4layer

    model.load_state_dict(torch.load(model_path,map_location=device))  # 加载模型参数
    files = os.listdir(info_dir)
    files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    if RANDOM and SELECT==False:
        num_to_select = int(len(files) * percentage / 100)
        selected_files = random.sample(files, num_to_select)
        files=selected_files
    elif RANDOM==False and SELECT:
        files=files[start_frame:end_frame]

    all_metrics = []
    connections_pred_count = 0
    gt_count = 0
    for file in files:
        file_path=info_dir + file
        image_path=image_dir + file.split(".")[0]+'.png'
        output_path=output_dir +model_name+"/"+ file.split(".")[0]+'_noline.png'

        inference_start_time=time.time()
        predicted_connections, connection_gt = predict_connections(model, file_path, threshold=0.9, device='cuda',distance_type=distance_type)
        inference_end_time=time.time()

        num_connections = len(predicted_connections)

        if output_image:
            draw_objects_and_lines(file_path, image_path, output_path, True, False)

        accuracy, precision, recall, f1, true_positives, false_positives, false_negatives,GED,NCA =connection_eval(predicted_connections,connection_gt,eval_extra)
        all_metrics.append((accuracy, precision, recall, f1, true_positives, false_positives, false_negatives,GED,NCA))

        connections_pred_count = connections_pred_count + len(true_positives)
        gt_count = gt_count + len(connection_gt)

        print("Detect the connections of file: {}".format(file))
        print("the num of Connections is: {}".format(num_connections))
        print("The inference time cost: {}ms".format((inference_end_time-inference_start_time)*1000) )
        for conn in predicted_connections:
            print(f"Object {conn[0]} is connected to Object {conn[1]}")

        print(f"Image Metrics:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, GED: {GED:.4f}, NCA: {NCA:.4f}")
        print("----")
        print()

    total_accuracy, total_precision, total_recall, total_f1,total_GED,total_NCA = calculate_total_metrics(all_metrics)
    print("########################################################")
    print("Image Num: {} ".format(len(files)))
    print("True Connections: {}".format(connections_pred_count))
    print("Ground Truth Num {}:".format(gt_count))
    print()
    print("Metrics:".format(len(files)))
    print(f"Accuracy: {total_accuracy:.4f}")
    print(f"Precision: {total_precision:.4f}")
    print(f"Recall: {total_recall:.4f}")
    print(f"F1-Score: {total_f1:.4f}")
    print(f"GED: {total_GED:.4f}")
    print(f"NCA: {total_NCA:.4f}")
    print("########################################################")
    metrics = {"Accuracy": total_accuracy, "Precision": total_precision, "Recall": total_recall, "F1-Score": total_f1,"GED":total_GED,"NCA":total_NCA}
    save_metrics_to_file(metrics, output_dir + model_name + "/metrics.txt")
