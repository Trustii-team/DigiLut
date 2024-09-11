import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
from sklearn.metrics import *
import shapely

def bpx_ratio(im):
    image = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # Define the dark threshold
    dark_threshold = 50
    
    # Count the number of dark pixels
    dark_pixels = np.sum(image <= dark_threshold)
    
    # Total number of pixels
    total_pixels = image.size
    
    # Calculate the percentage of dark pixels
    dark_percentage = (dark_pixels / total_pixels)
    
    return dark_percentage

def wpx_ratio(im):
    image = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # Define the bright threshold
    bright_threshold = 200
    
    # Count the number of bright pixels
    bright_pixels = np.sum(image >= bright_threshold)
    
    # Total number of pixels
    total_pixels = image.size
    
    # Calculate the percentage of bright pixels
    bright_percentage = (bright_pixels / total_pixels)
    
    return bright_percentage

def train_model(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    _, _, f2, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', beta=2)
    return epoch_loss, epoch_accuracy, precision, recall, f1, f2
    
def evaluate_model(model, test_loader, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    _, _, f2, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', beta=2)
    return epoch_loss, epoch_accuracy, precision, recall, f1, f2

def generate_preds(model, test_loader, device):
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return all_labels, all_predictions

def arith_mean(probs):
    mean_probs = torch.mean(probs, dim=0)
    _, final_arith_preds = torch.max(mean_probs, 1)
    return final_arith_preds

def geo_mean(probs):
    geo_mean_probs = torch.exp(torch.mean(torch.log(probs), dim=0))
    _, final_geo_preds = torch.max(geo_mean_probs, 1)
    return final_geo_preds

def harm_mean(probs):
    harm_mean_probs = torch.mean(1.0 / probs, dim=0)
    harm_mean_probs = 1.0 / harm_mean_probs
    harm_mean_probs = harm_mean_probs / harm_mean_probs.sum(dim=1, keepdim=True)
    _, final_harm_preds = torch.max(harm_mean_probs, 1)
    return final_harm_preds

def majority_voting(preds):
    majority_preds, _ = mode(preds, axis=0)
    return majority_preds

# Detect available GPUs
def get_available_devices():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        devices = [torch.device('cpu')]
        print("No GPUs available, using CPU")
    return devices

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

def build_model(MODEL_PATH, device):
    EXTRACT_FEATURES = False # model = feature extractor if true, model = classifier if false
    NUM_CLASSES = 2  # only used if EXTRACT_FEATURES = False
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(MODEL_PATH, map_location=device)

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    model = load_model_weights(model, state_dict)
    for p in model.parameters():
        p.requires_grad = False

    if EXTRACT_FEATURES:
        model.fc = torch.nn.Sequential()
    else:
        model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 128),
                                       torch.nn.BatchNorm1d(128),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(128, 32),
                                       torch.nn.BatchNorm1d(32),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(32, NUM_CLASSES),
                                       torch.nn.Softmax(dim=1))
    return model

def generate_preds_probas(model, test_loader, device):
    model.eval()
    all_preds, all_probas = [], []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            all_probas.append(outputs.cpu().numpy())
    all_preds, all_probas = np.concatenate(all_preds), np.concatenate(all_probas)
    return all_preds, all_probas


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-5, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            self.trace_func(f'test loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss
        
def get_pred(pred_csv, slide):
    coords = np.array([[int(l[2]), int(l[3].split(".")[0])] for l in pred_csv[pred_csv.filename == slide].image.str.split("_").values])
    pred = pred_csv[pred_csv.filename == slide].preds.values
    return coords, pred

def retrieve_rois(pos_coords, patch_size, nb_roi):
    n_intersections = 1
    patches = np.array([shapely.box(x, y, x+patch_size, y+patch_size, ccw=False) for x,y in pos_coords])
    while n_intersections > 0 and len(patches) > nb_roi:
        tree = shapely.STRtree(patches)
        new_patches = []
        resolved_indices = []
        n_intersections = 0
        for i in range(len(patches)):
            if i in resolved_indices:
                continue
            else:
                intersection = tree.query_nearest(patches[i], max_distance=1, exclusive=True, return_distance=False)
                if len(intersection) > 0:
                    new_patches.append(shapely.unary_union(patches[[i] + list(intersection)]))
                    resolved_indices.extend([i] + list(intersection))
                    n_intersections +=1
                else:
                    new_patches.append(patches[i])
                    resolved_indices.append(i)
        patches = np.array(new_patches)
    return patches
