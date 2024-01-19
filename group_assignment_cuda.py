# -*- coding: utf-8 -*-
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, LongTensor
from typing import Tuple, List, Callable
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools

# function that removes final _ separated substring from string
def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

# min-max scaling
def minmax(trial):
    min = trial.min()
    max = trial.max()
    scaled_trial = (trial - min)/(max-min)
    return scaled_trial

#Z-score normalisation OPTIONAL
def zscore(trial):
    mean = np.mean(trial, axis=0)
    std = np.std(trial, axis=0)
    scaled_trial = (trial - mean) / std
    return scaled_trial

# robust scaling
def robust(trial):
    median = np.median(trial, axis=0)
    q75, q25 = np.percentile(trial, [75, 25], axis=0)
    iqr = q75 - q25
    scaled_trial = (trial - median) / iqr
    return scaled_trial

# clipping min and max values
def clip(trial, a_min=-3.5, a_max=3.5):
    clipped_trial = np.clip(trial, a_min, a_max)
    return clipped_trial

#downsamples data by totaltimesteps/factor
def downsample(trial, factor):
    ds_trial = trial[:,::factor]
    return ds_trial

# Function that transforms a 1D feature vector into a 2D map of the activitiy on the scalp
def array_to_mesh(arr):
    
    input_rows = 20
    input_columns = 21
    input_channels = 248

    assert arr.shape == (input_channels,),"the shape of the input array should be (248,) because there are 248 MEG channels,received array of shape " + str(arr.shape)
    output = np.zeros((input_rows,input_columns),dtype = float)
    
    #121
    output[0][10] = arr[120]
      
    #89
    output[1][12] = arr[151]
    output[1][11] = arr[119]
    output[1][10] = arr[88]
    output[1][9] = arr[89]
    output[1][8] = arr[121]
    
    #61
    output[2][13] = arr[150]
    output[2][12] = arr[118]
    output[2][11] = arr[87]
    output[2][10] = arr[60]
    output[2][9] = arr[61]
    output[2][8] = arr[90]
    output[2][7] = arr[122]
    
    #37
    output[3][14] = arr[149]
    output[3][13] = arr[117]
    output[3][12] = arr[86]
    output[3][11] = arr[59]
    output[3][10] = arr[36]
    output[3][9] = arr[37]
    output[3][8] = arr[62]
    output[3][7] = arr[91]
    output[3][6] = arr[123]
    
    #19
    output[4][17] = arr[194]
    output[4][16] = arr[175]
    output[4][15] = arr[148]
    output[4][14] = arr[116]
    output[4][13] = arr[85]
    output[4][12] = arr[58]
    output[4][11] = arr[35]
    output[4][10] = arr[18]
    output[4][9] = arr[19]
    output[4][8] = arr[38]
    output[4][7] = arr[63]
    output[4][6] = arr[92]
    output[4][5] = arr[152]
    output[4][4] = arr[176]

    #5
    output[5][20] = arr[247]
    output[5][19] = arr[227]
    output[5][18] = arr[193]
    output[5][17] = arr[174]
    output[5][16] = arr[147]
    output[5][15] = arr[115]
    output[5][14] = arr[84]
    output[5][13] = arr[57]
    output[5][12] = arr[34]
    output[5][11] = arr[17]
    output[5][10] = arr[4]
    output[5][9] = arr[5]
    output[5][8] = arr[20]
    output[5][7] = arr[39]
    output[5][6] = arr[64]
    output[5][5] = arr[93]
    output[5][4] = arr[125]
    output[5][3] = arr[153]
    output[5][2] = arr[177]
    output[5][1] = arr[211]
    output[5][0] = arr[228]
    
    #4
    output[6][20] = arr[246]
    output[6][19] = arr[226]
    output[6][18] = arr[192]
    output[6][17] = arr[173]
    output[6][16] = arr[146]
    output[6][15] = arr[114]
    output[6][14] = arr[83]
    output[6][13] = arr[56]
    output[6][12] = arr[33]
    output[6][11] = arr[16]
    output[6][10] = arr[3]
    output[6][9] = arr[6]
    output[6][8] = arr[21]
    output[6][7] = arr[40]
    output[6][6] = arr[65]
    output[6][5] = arr[94]
    output[6][4] = arr[126]
    output[6][3] = arr[154]
    output[6][2] = arr[178]
    output[6][1] = arr[212]
    output[6][0] = arr[229]

    
    #3
    output[7][19] = arr[245]
    output[7][18] = arr[210]
    output[7][17] = arr[172]
    output[7][16] = arr[145]
    output[7][15] = arr[113]
    output[7][14] = arr[82]
    output[7][13] = arr[55]
    output[7][12] = arr[32]
    output[7][11] = arr[15]
    output[7][10] = arr[2]
    output[7][9] = arr[7]
    output[7][8] = arr[22]
    output[7][7] = arr[41]
    output[7][6] = arr[66]
    output[7][5] = arr[95]
    output[7][4] = arr[127]
    output[7][3] = arr[155]
    output[7][2] = arr[195]
    output[7][1] = arr[230]
            
    #8
    output[8][19] = arr[244]
    output[8][18] = arr[209]
    output[8][17] = arr[171]
    output[8][16] = arr[144]
    output[8][15] = arr[112]
    output[8][14] = arr[81]
    output[8][13] = arr[54]
    output[8][12] = arr[31]
    output[8][11] = arr[14]
    output[8][10] = arr[1]
    output[8][9] = arr[8]
    output[8][8] = arr[23]
    output[8][7] = arr[42]
    output[8][6] = arr[67]
    output[8][5] = arr[96]
    output[8][4] = arr[128]
    output[8][3] = arr[156]
    output[8][2] = arr[196]
    output[8][1] = arr[231]
    
    #1
    output[9][19] = arr[243]
    output[9][18] = arr[208]
    output[9][17] = arr[170]
    output[9][16] = arr[143]
    output[9][15] = arr[111]
    output[9][14] = arr[80]
    output[9][13] = arr[53]
    output[9][12] = arr[30]
    output[9][11] = arr[13]
    output[9][10] = arr[0]
    output[9][9] = arr[9]
    output[9][8] = arr[24]
    output[9][7] = arr[43]
    output[9][6] = arr[68]
    output[9][5] = arr[97]
    output[9][4] = arr[129]
    output[9][3] = arr[157]
    output[9][2] = arr[197]
    output[9][1] = arr[232]
    
    #12
    output[10][18] = arr[225]
    output[10][17] = arr[191]
    output[10][16] = arr[142]
    output[10][15] = arr[110]
    output[10][14] = arr[79]
    output[10][13] = arr[52]
    output[10][12] = arr[29]
    output[10][11] = arr[12]
    output[10][10] = arr[11]
    output[10][9] = arr[10]
    output[10][8] = arr[25]
    output[10][7] = arr[44]
    output[10][6] = arr[69]
    output[10][5] = arr[98]
    output[10][4] = arr[130]
    output[10][3] = arr[179]
    output[10][2] = arr[213]
    
    #28
    output[11][16] = arr[169]
    output[11][15] = arr[141]
    output[11][14] = arr[109]
    output[11][13] = arr[78]
    output[11][12] = arr[51]
    output[11][11] = arr[28]
    output[11][10] = arr[27]
    output[11][9] = arr[26]
    output[11][8] = arr[45]
    output[11][7] = arr[70]
    output[11][6] = arr[99]
    output[11][5] = arr[131]
    output[11][4] = arr[158]
    
    #49
    output[12][17] = arr[190]
    output[12][16] = arr[168]
    output[12][15] = arr[140]
    output[12][14] = arr[108]
    output[12][13] = arr[77]
    output[12][12] = arr[50]
    output[12][11] = arr[49]
    output[12][10] = arr[48]
    output[12][9] = arr[47]
    output[12][8] = arr[46]
    output[12][7] = arr[71]
    output[12][6] = arr[100]
    output[12][5] = arr[132]
    output[12][4] = arr[159]
    output[12][3] = arr[180]

    
    #75
    output[13][18] = arr[224]
    output[13][17] = arr[207]
    output[13][16] = arr[189]
    output[13][15] = arr[167]
    output[13][14] = arr[139]
    output[13][13] = arr[107]
    output[13][12] = arr[76]
    output[13][11] = arr[75]
    output[13][10] = arr[74]
    output[13][9] = arr[73]
    output[13][8] = arr[72]
    output[13][7] = arr[101]
    output[13][6] = arr[133]
    output[13][5] = arr[160]
    output[13][4] = arr[181]
    output[13][3] = arr[198]
    output[13][2] = arr[214]
    
    #105
    output[14][18] = arr[242]
    output[14][17] = arr[223]
    output[14][16] = arr[206]
    output[14][15] = arr[188]
    output[14][14] = arr[166]
    output[14][13] = arr[138]
    output[14][12] = arr[106]
    output[14][11] = arr[105]
    output[14][10] = arr[104]
    output[14][9] = arr[103]
    output[14][8] = arr[102]
    output[14][7] = arr[134]
    output[14][6] = arr[161]
    output[14][5] = arr[182]
    output[14][4] = arr[199]
    output[14][3] = arr[215]
    output[14][2] = arr[233]
    
    
    #137
    output[15][16] = arr[241]
    output[15][15] = arr[222]
    output[15][14] = arr[205]
    output[15][13] = arr[187]
    output[15][12] = arr[165]
    output[15][11] = arr[137]
    output[15][10] = arr[136]
    output[15][9] = arr[135]
    output[15][8] = arr[162]
    output[15][7] = arr[183]
    output[15][6] = arr[200]
    output[15][5] = arr[216]
    output[15][4] = arr[234]
    
    
    #mix
    output[16][15] = arr[240]
    output[16][14] = arr[221]
    output[16][13] = arr[204]
    output[16][12] = arr[186]
    output[16][11] = arr[164]
    output[16][10] = arr[163]
    output[16][9] = arr[184]
    output[16][8] = arr[201]
    output[16][7] = arr[217]
    output[16][6] = arr[235]
   
    #186
    output[17][12] = arr[220]
    output[17][11] = arr[203]
    output[17][10] = arr[185]
    output[17][9] = arr[202]
    output[17][8] = arr[218]
   
    #220
    output[18][11] = arr[239]
    output[18][10] = arr[219]
    #output[18][9] = arr[236] # Is often an outlier.. Perhaps broken sensor
    
    #mix
    output[19][11] = arr[238]
    output[19][10] = arr[237]
    
    return output

# Code for storing data in a folder into an array
def preprocess_files(path='Final Project data/Cross/train', downsampling=30, mesh=True):
    label_to_int = {'rest': 0, 'task_motor': 1, 'task_story_math': 2, 'task_working_memory': 3}

    cross_data_train = [] # Store data
    cross_data_train_labels = [] # Store labels (based on filename)

    files = os.listdir(path)

    for file in files:
        file_path = f'{path}/{file}'
        
        with h5py.File(file_path, 'r') as h5_file:
            # obtain labels
            dataset_name = get_dataset_name(file_path)
            label = get_dataset_name(dataset_name)
            cross_data_train_labels.append(label_to_int[label])
            
            # obtain X_data
            matrix = h5_file.get(dataset_name)[()]
            normalisedMatrix = downsample(zscore(matrix), downsampling) # apply minmax normalisation and downsampling
            cross_data_train.append(normalisedMatrix.T) # Transpose
            
    # Convert lists with data to arrays
    X_array = np.array(cross_data_train)
    y_array = np.array(cross_data_train_labels)
    
    if mesh:
        # convert the 1D feature vectors into 2D maps of the brain scans
        X_mesh_array = np.apply_along_axis(array_to_mesh, axis=-1, arr=X_array)
        X = torch.from_numpy(X_mesh_array).float().unsqueeze(2)
    else:
        # Else just 1d vectors
        X = torch.from_numpy(X_array).float()
    
    y = torch.tensor(y_array).long()
    return X, y

# train batch
def train_batch(
        network: torch.nn.Module,
        X_batch: FloatTensor,
        y_batch: LongTensor,
        loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
        optimizer: torch.optim.Optimizer
        ) -> float: # batch loss
    # set model to training mode
    network.train()
    # train network
    batch_prediction = network(X_batch).float() # Forward pass
    # calculate loss
    batch_loss = loss_fn(batch_prediction, y_batch)
    # Calculate gradients
    batch_loss.backward()
    # Back propagate and reset gradients
    optimizer.step()
    optimizer.zero_grad()
    
    return batch_loss.item()

# train all batches in dataloader
def train_epoch(
        network: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor],
        optimizer: torch.optim.Optimizer
        ) -> float: # epoch loss
    # Set initial loss value to 0
    loss = 0
    # Train all the batches in the dataloader
    for index, (X_batch, y_batch) in enumerate(dataloader):
        loss += train_batch(network=network, X_batch=X_batch, y_batch=y_batch, loss_fn=loss_fn, optimizer=optimizer)
    # divide loss by number of batches
    loss /= (index + 1)
    
    return loss

# evaluate batch with test data
def eval_batch(
        network: torch.nn.Module,
        X_batch: FloatTensor,
        y_batch: LongTensor,
        loss_fn: Callable[[FloatTensor, LongTensor], FloatTensor]
        ) -> float: # batch loss
    # set model to evaluation mode
    network.eval()
    # no need to track gradients
    with torch.no_grad():
        # calculate loss on prediction
        batch_prediction = network(X_batch).float()
        batch_loss = loss_fn(batch_prediction, y_batch)
        
    return batch_loss.item()

# evaluate epoch with test data
def eval_epoch(
        network: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[FloatTensor, LongTensor], FloatTensor]
        ) -> float: #epoch's loss
    # Set the initial loss value.
    loss = 0.
    # Iterate over the batches in the dataloader
    for index, (X_batch, y_batch) in enumerate(dataloader):
        # Compute loss
        loss += eval_batch(network=network, X_batch=X_batch, y_batch=y_batch, loss_fn=loss_fn)
    # divide loss by number of batches
    loss /= (index + 1)
    
    return loss

# computes accuracy of one batch
def accuracy_batch(
        network: torch.nn.Module,
        X_batch: FloatTensor,
        y_batch: LongTensor
        ) -> float:
    # Set model to evaluation mode
    network.eval()
    # dont track gradients
    with torch.no_grad():
        # compute prediction
        prediction_batch = network(X_batch).detach().cpu().numpy()
        y_pred = np.argmax(prediction_batch, axis=1)
        # cast true labels to numpy
        y_true = y_batch.cpu().numpy()
        accuracy_batch = accuracy_score(y_true, y_pred)
    
    return accuracy_batch
     
# Compute accuracy over all batches
def accuracy_epoch(
        network: torch.nn.Module,
        dataloader: DataLoader
        ) -> (float, float, float, float):
    # Set initial accuracy to 0
    accuracy = 0
    # calculate accuracy of all batches
    for index, (X_batch, y_batch) in enumerate(dataloader):
        # compute accuracy for batch
        accuracy += accuracy_batch(network=network, X_batch=X_batch, y_batch=y_batch)
    # divide accuracy by amount of batches
    accuracy /= (index + 1)
    
    return accuracy

def generate_combinations(param_dict):
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())

    # Generate all possible combinations of parameter values
    param_combinations = list(itertools.product(*param_values))

    # Create dictionaries for each combination
    result_dicts = []
    for combo in param_combinations:
        result_dict = dict(zip(param_names, combo))
        result_dicts.append(result_dict)

    return result_dicts

def plot_graphs(
        train_losses: list,
        val_losses: list,
        train_accuracies: list,
        val_accuracies: list
        ):
    # plots!
    fig, axs = plt.subplots(1, 2, figsize=(12,4))

    # Your loss plotting here: x-axis for epochs and y-axis for train and validation losses

    axs[0].plot(train_losses, 'r-', label='Train Loss')
    axs[0].plot(val_losses, 'b-', label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Losses')
    axs[0].set_ylim(0, 1.5)
    axs[0].legend()

    # Your accuracy plotting here: x-axis for epochs and y-axis for train and validation accuracies

    axs[1].plot(train_accuracies, 'g-', label='Train Accuracy')
    axs[1].plot(val_accuracies, 'y-', label='Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracies')
    axs[1].set_ylim(0.25, 1)
    axs[1].legend()
    return

def train_model(
        model: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        parameters: dict,
        learning_rate: float=0.001,
        epochs: int=30,
        report: bool=False
        ):
    # Set seed
    torch.manual_seed(0)

    # Create model
    if model == 'CLSTM':
        model = CLSTM(**parameters).cuda()
    elif model == 'RNN':
        model = RNN(**parameters).cuda()
    elif model == 'CRNN':
        model = CRNN(**parameters).cuda()
        
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for t in range(epochs):
        train_loss = train_epoch(network=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        val_loss = eval_epoch(network=model, dataloader=val_dataloader, loss_fn=loss_fn)
        train_accuracy = accuracy_epoch(network=model, dataloader=train_dataloader)
        val_accuracy = accuracy_epoch(network=model, dataloader=val_dataloader)

        if report:
            print("Epoch {}".format(t))
            print(" Training Loss: {}".format(train_loss))
            print(" Validation Loss: {}".format(val_loss))
            print(" Training Accuracy: {}".format(train_accuracy))
            print(" Validation Accuracy: {}".format(val_accuracy))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    
    max_index, max_accuracy = max(enumerate(val_accuracies), key=lambda x: x[1])
    print("Highest accuracy: ",max_accuracy, " at epoch: ", max_index+1)
    return train_losses, val_losses, train_accuracies, val_accuracies, max_index, max_accuracy, model
    

def grid_search(
        model: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        param_grid: dict,
        learning_rates: list,
        ):
    # Generate all possible combinations from the parameter grid
    param_combinations = generate_combinations(param_grid)
    
    print(param_combinations)
    
    number_of_combos = len(param_combinations)
    number_of_lr = len(learning_rates)
    number_of_models = number_of_combos * number_of_lr
    
    print(f"Trying {number_of_models} different models")
    
    best_accuracy = 0
    
    # for each param_grid and learning rate train the model and evaluate
    for learning_rate in learning_rates:
        for parameters in param_combinations:
            train_losses, val_losses, train_accuracies, val_accuracies, max_index, current_accuracy, model = train_model(
                model,
                train_dataloader,
                val_dataloader,
                parameters,
                learning_rate
                )
            # if best accuracy is beaten set new best accuracy and save parameters
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_parameters = parameters
                best_epoch = max_index+1
                best_learning_rate = learning_rate
    
    # Report results
    print("=====================================")
    print("The ultimate model has an accuracy of: ", best_accuracy)
    print("Using the following parameters: ", best_parameters)
    print("And the following learning rate: ", best_learning_rate)
    print("At epoch: ", best_epoch)
    print("=====================================")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Class for creating a 2d mesh dataset 
class MEGMeshDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# Convolutional LSTM model
class CLSTM(nn.Module):
    def __init__(
            self,
            num_filters=1,
            kernel_size=1,
            hidden_size=256,
            num_classes=4,
            num_rows=20,
            num_columns=21,
            dropout_rate=0.50
            ):
        super(CLSTM, self).__init__()
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Calculate the number of features after the second convolutional layer
        num_features_after_conv = self._calculate_conv_features(1, num_rows, num_columns)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_features_after_conv, hidden_size=hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Fully connected layer
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        # Softmax activation
        self.softmax = nn.Softmax(dim=1)
        
    def _calculate_conv_features(self, input_channels, height, width):
        # Define a dummy input with batch size 1
        dummy_input = torch.randn(1, input_channels, height, width)

        # Pass the input through the first two convolutional layers
        dummy_output = self.conv(dummy_input)
        
        print(dummy_output.shape)

        # Calculate the number of features after flattening
        num_features_after_conv = dummy_output.view(dummy_output.size(0), -1).size(1)

        return num_features_after_conv
        
    def forward(self, X):
        # Get the dimensions of the input of shape (batch_size, sequence_length, input_streams, num_rows, num_columns)
        batch_size, sequence_length, input_streams, num_rows, num_columns = X.shape
        giant_batch_size = batch_size * sequence_length
        
        # Reshape the separate input batches into one long batch ((batch_size * sequence_length), input_streams, num_rows, num_columns) so that it can be input to the conv layers. (I_c1 := Input_convolutional_1)
        X = X.view(giant_batch_size, input_streams, num_rows, num_columns)

        # Feed giant batch through all the convolutional layers
        # Conv layer 1
        X = self.conv(X)
        X = self.relu(X) # ReLU activate
        X = self.dropout1(X)

        # Restore the batch and sequence distinction and reshape to (batch_size, sequence_length, input_size) shape
        X = X.view(batch_size, sequence_length, -1) # Reshape for the fully connected layer
        
        # Feed through the LSTM
        X, _ = self.lstm(X)
        X = self.dropout2(X)

        # Second fully connected layer
        X = self.linear(X)
        Y = self.dropout3(X)
        
        # Return classification at final timestep
        return Y[:,-1,:]
    
# Convolutional RNN model
class CRNN(nn.Module):
    def __init__(
            self,
            num_filters=1,
            kernel_size=1,
            hidden_size=256,
            num_classes=4,
            num_rows=20,
            num_columns=21,
            dropout_rate=0.50
            ):
        super(CRNN, self).__init__()
        # First convolutional layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Calculate the number of features after the second convolutional layer
        num_features_after_conv = self._calculate_conv_features(1, num_rows, num_columns)
        
        # LSTM layer
        self.rnn = nn.RNN(input_size=num_features_after_conv, hidden_size=hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Fully connected layer
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        # Softmax activation
        self.softmax = nn.Softmax(dim=1)
        
    def _calculate_conv_features(self, input_channels, height, width):
        # Define a dummy input with batch size 1
        dummy_input = torch.randn(1, input_channels, height, width)

        # Pass the input through the first two convolutional layers
        dummy_output = self.conv(dummy_input)
        
        print(dummy_output.shape)

        # Calculate the number of features after flattening
        num_features_after_conv = dummy_output.view(dummy_output.size(0), -1).size(1)

        return num_features_after_conv
        
    def forward(self, X):
        # Get the dimensions of the input of shape (batch_size, sequence_length, input_streams, num_rows, num_columns)
        batch_size, sequence_length, input_streams, num_rows, num_columns = X.shape
        giant_batch_size = batch_size * sequence_length
        
        # Reshape the separate input batches into one long batch ((batch_size * sequence_length), input_streams, num_rows, num_columns) so that it can be input to the conv layers. (I_c1 := Input_convolutional_1)
        X = X.view(giant_batch_size, input_streams, num_rows, num_columns)

        # Feed giant batch through all the convolutional layers
        # Conv layer 1
        X = self.conv(X)
        X = self.relu(X) # ReLU activate
        X = self.dropout1(X)

        # Restore the batch and sequence distinction and reshape to (batch_size, sequence_length, input_size) shape
        X = X.view(batch_size, sequence_length, -1) # Reshape for the fully connected layer
        
        # Feed through the LSTM
        X, _ = self.rnn(X)
        X = self.dropout2(X)

        # Second fully connected layer
        X = self.linear(X)
        Y = self.dropout3(X)
        
        # Return classification at final timestep
        return Y[:,-1,:]
    
class RNN(nn.Module):
    def __init__(
            self,
            input_size=248,
            hidden_size=256,
            output_size=4
            ):
        super(RNN, self).__init__()
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, X):
        # Forward pass through RNN
        X, _ = self.rnn(X)
        # Only take the output from the final time step
        Y = self.fc(X)
        
        # return classificiation at final timestep
        return Y[:,-1,:]
    

'''
Training and testing models on the Cross participant training and testing set
Reporting results and saving training data for plotting
Training final models on the reported hyperparameters
'''
# =============================================================================
# Preprocessing 1D data
# =============================================================================

# Preprocess all the files. Should result in 64 different data samples   
X_train, y_train = preprocess_files(mesh=False)
X_test, y_test = preprocess_files(path='Final Project data/Cross/test', mesh=False)

# Move data to GPU
X_train = X_train.cuda()
y_train = y_train.cuda()

X_test = X_test.cuda()
y_test = y_test.cuda()

# Create a custom dataset for the dataloader
train_dataset = MEGMeshDataset(X_train, y_train)
test_dataset = MEGMeshDataset(X_test, y_test)

# Create a dataloader in order for creating batches and shuffling
train_dataloader_cross = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader_cross = DataLoader(test_dataset, batch_size=6, shuffle=True)

# =============================================================================
# Preprocessing 2D data
# =============================================================================

# Preprocess all the files. Should result in 64 different data samples   
X_train_2D, y_train = preprocess_files()
X_test_2D, y_test = preprocess_files(path='Final Project data/Cross/test')

# Move data to GPU
X_train_2D = X_train_2D.cuda()
y_train = y_train.cuda()

X_test_2D = X_test_2D.cuda()
y_test = y_test.cuda()

# Create a custom dataset for the dataloader
train_dataset_2D = MEGMeshDataset(X_train_2D, y_train)
test_dataset_2D = MEGMeshDataset(X_test_2D, y_test)

# Create a dataloader in order for creating batches and shuffling
train_dataloader_2D_cross = DataLoader(train_dataset_2D, batch_size=8, shuffle=True)
val_dataloader_2D_cross = DataLoader(test_dataset_2D, batch_size=6, shuffle=True)

# =============================================================================
# Grid Search
# =============================================================================

param_grid_clstm = {
    'num_filters': [1],
    'kernel_size': [1],
    'hidden_size': [256]
    }
param_grid_rnn = {
    'hidden_size': [64]
    }

train_losses_clstm_cross, val_losses_clstm_cross, train_accuracies_clstm_cross, val_accuracies_clstm_cross = grid_search(
    model='CLSTM',
    train_dataloader=train_dataloader_2D_cross,
    val_dataloader=val_dataloader_2D_cross,
    param_grid=param_grid_clstm,
    learning_rates=[0.001]
    )

train_losses_crnn_cross, val_losses_crnn_cross, train_accuracies_crnn_cross, val_accuracies_crnn_cross = grid_search(
    model='CRNN',
    train_dataloader=train_dataloader_2D_cross,
    val_dataloader=val_dataloader_2D_cross,
    param_grid=param_grid_clstm,
    learning_rates=[0.001]
    )

train_losses_rnn_cross, val_losses_rnn_cross, train_accuracies_rnn_cross, val_accuracies_rnn_cross = grid_search(
    model='RNN',
    train_dataloader=train_dataloader_cross,
    val_dataloader=val_dataloader_cross,
    param_grid=param_grid_rnn,
    learning_rates=[0.001]
    )

'''
============CLSTM====================
The ultimate model has an accuracy of:  0.8541666666666665
Using the following parameters:  {'num_filters': 1, 'kernel_size': 1, 'hidden_size': 256}
And the following learning rate:  0.001
At epoch:  14
============CRNN=====================
The ultimate model has an accuracy of:  0.8541666666666666
Using the following parameters:  {'num_filters': 1, 'kernel_size': 1, 'hidden_size': 256}
And the following learning rate:  0.001
At epoch:  18
=============RNN=====================
The ultimate model has an accuracy of:  0.7291666666666667
Using the following parameters:  {'hidden_size': 64}
And the following learning rate:  0.001
At epoch:  10
=====================================
'''

'''
============================================================================================================================================================
Training models on the intra data and testing on this data
Reporting the best results on that data
============================================================================================================================================================
'''

# =============================================================================
# Preprocessing 1D data
# =============================================================================

# Preprocess all the files. Should result in 64 different data samples   
X_train, y_train = preprocess_files(path='Final Project data/Intra/train', mesh=False)
X_test, y_test = preprocess_files(path='Final Project data/Intra/test', mesh=False)

# Move data to GPU
X_train = X_train.cuda()
y_train = y_train.cuda()

X_test = X_test.cuda()
y_test = y_test.cuda()

# Create a custom dataset for the dataloader
train_dataset = MEGMeshDataset(X_train, y_train)
test_dataset = MEGMeshDataset(X_test, y_test)

# Create a dataloader in order for creating batches and shuffling
train_dataloader_intra = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader_intra = DataLoader(test_dataset, batch_size=6, shuffle=True)

# =============================================================================
# Preprocessing 2D data
# =============================================================================

# Preprocess all the files. Should result in 64 different data samples   
X_train_2D, y_train = preprocess_files(path='Final Project data/Intra/train')
X_test_2D, y_test = preprocess_files(path='Final Project data/Intra/test')

# Move data to GPU
X_train_2D = X_train_2D.cuda()
y_train = y_train.cuda()

X_test_2D = X_test_2D.cuda()
y_test = y_test.cuda()

# Create a custom dataset for the dataloader
train_dataset_2D = MEGMeshDataset(X_train_2D, y_train)
test_dataset_2D = MEGMeshDataset(X_test_2D, y_test)

# Create a dataloader in order for creating batches and shuffling
train_dataloader_2D_intra = DataLoader(train_dataset_2D, batch_size=4, shuffle=True)
val_dataloader_2D_intra = DataLoader(test_dataset_2D, batch_size=2, shuffle=True)

# =============================================================================
# Grid Search
# =============================================================================

param_grid_clstm = {
    'num_filters': [1],
    'kernel_size': [3],
    'hidden_size': [128]
    }
param_grid_rnn = {
    'hidden_size': [128]
    }

train_losses_clstm_intra, val_losses_clstm_intra, train_accuracies_clstm_intra, val_accuracies_clstm_intra = grid_search(
    model='CLSTM',
    train_dataloader=train_dataloader_2D_intra,
    val_dataloader=val_dataloader_2D_intra,
    param_grid=param_grid_clstm,
    learning_rates=[0.001]
    )

train_losses_crnn_intra, val_losses_crnn_intra, train_accuracies_crnn_intra, val_accuracies_crnn_intra = grid_search(
    model='CRNN',
    train_dataloader=train_dataloader_2D_intra,
    val_dataloader=val_dataloader_2D_intra,
    param_grid=param_grid_clstm,
    learning_rates=[0.001]
    )

train_losses_rnn_intra, val_losses_rnn_intra, train_accuracies_rnn_intra, val_accuracies_rnn_intra = grid_search(
    model='RNN',
    train_dataloader=train_dataloader_intra,
    val_dataloader=val_dataloader_intra,
    param_grid=param_grid_rnn,
    learning_rates=[0.001]
    )

'''
============CLSTM====================
The ultimate model has an accuracy of:  1.0
Using the following parameters:  {'num_filters': 1, 'kernel_size': 3, 'hidden_size': 128}
And the following learning rate:  0.001
At epoch:  1
============CRNN=====================
The ultimate model has an accuracy of:  1.0
Using the following parameters:  {'num_filters': 1, 'kernel_size': 3, 'hidden_size': 128}
And the following learning rate:  0.001
At epoch:  3
=============RNN=====================
The ultimate model has an accuracy of:  1.0
Using the following parameters:  {'hidden_size': 128}
And the following learning rate:  0.001
At epoch:  1
=====================================
'''



