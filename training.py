import argparse
import json
import logging
import os
import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import glob
import subprocess as sb
import boto3

sb.call([sys.executable, "-m", "pip", "install", 'scikit-learn'])
sb.call([sys.executable, "-m", "pip", "install", 'boto3'])
hidden_size = 100

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

def _load_data(data, n_prev=50):
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        if i == 0:
            continue
        docX.append(data.iloc[i - 1:i + n_prev - 1].values)
        docY.append(data.iloc[i + n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


# 学習用とテスト用データを分割、ただし分割する際に_load_data()を適用
def train_test_split(df, test_size=0.1, n_prev=50):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)


def preprocessing(train_dir, dataset, is_x):
    if dataset.ndim != 2:
        dataset = dataset.reshape(-1, 1)
    scaler = joblib.load(os.path.join(train_dir, 'scaler.save'))
    dataset_src = scaler.transform(np.log(dataset))
    if is_x:
        return dataset_src.reshape(-1, 24, 1).tolist()

    else:
        return dataset_src.tolist()


# Based LSTM Network
class Net(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Net, self).__init__()

        self.module = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.module(inputs, hidden0) #LSTM層
        output = self.output_layer(output[:, -1, :]) #全結合層

        return output

    
def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t)


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args):
    
    batch_size = args.batch_size
    epochs_num = args.epochs
    train_dir = args.train
#     code_dir = os.path.join(args.model_dir, 'code')
#     os.mkdir(code_dir)
    channel_name = 'training'
    
#     input_data_files = ["dataset/mnist/inference.py", "dataset/mnist/scaler.save", "dataset/mnist/requirements.txt"]
#     for input_data_file in input_data_files:
#         download(code_dir, input_data_file)

    # load data

    # Take the set of files and read them all into a single pandas dataframe
    # input_files = [os.path.join(train_dir, file)
    #                for file in os.listdir(train_dir)]

    # if len(input_files) == 0:
    #     raise ValueError(('There are no files in {}.\n' +
    #                       'This usually indicates that the channel ({}) was incorrectly specified,\n' +
    #                       'the data specification in S3 was incorrectly specified or the role specified\n' +
    #                       'does not have permission to access the data.').format(train_dir, channel_name))
    
    input_files = glob.glob("{}/*.csv".format(train_dir))
    
    raw_data = [ pd.read_csv(file) for file in input_files ]
    train_data = pd.concat(raw_data)

    length_of_sequences = 24
    in_out_neurons = 1

    (x_train, y_train), (x_test, y_test) = train_test_split(
        train_data[["demand"]], test_size=0.2, n_prev=length_of_sequences)

    train_x = preprocessing(train_dir, x_train, True)
    train_t = preprocessing(train_dir, y_train, False)
    test_x = preprocessing(train_dir, x_test, True)
    test_t = preprocessing(train_dir, y_test, False)
    
    training_size = x_train.shape[0]
    test_size = x_test.shape[0]

    

    # set the seed for generating random numbers
#     torch.manual_seed(args.seed)
    
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net(1, hidden_size, 1).to(device)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = mkRandomBatch(train_x, train_t, batch_size)
            
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))
    
    save_model(model, args.model_dir)


def test(model, test_x, test_t, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in zip(test_x, test_t):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_x)
    printw('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_x),
        100. * correct / len(test_x)))

def input_fn(request_body, request_content_type):
    s3 = boto3.resource('s3', aws_access_key_id="AKIAXAGPD6STVODIEREU", aws_secret_access_key="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")
    bucket_name = "sagemaker-ap-northeast-1-481470706855"
    file = "dataset/mnist/scaler.save"
    content_object = s3.Object(bucket_name, file)
    s3.Bucket(bucket_name).download_file(file, 'scaler.save')
    
    if request_content_type == 'application/json':
    # pass through json (assumes it's correctly formed)
#         d = request_body.read().decode('utf-8')
        try:
            scaler = joblib.load('scaler.save')
            d = json.loads(request_body)
            d = np.array(d)

            d = d.reshape(-1, 1)
            
            dataset_sc = scaler.transform(np.log(d)).reshape(1, -1, 1)
            return torch.Tensor(dataset_sc)
        except:
            raise ValueError('{{"error": "could not preprocess input data: {}"}}'.format(request_body))

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(context.request_content_type or "unknown"))
    
    
def output_fn(prediction, content_type):
#     raise ValueError('{{"prediction": {}, "type": {}}}'.format(prediction, type(prediction)))
        
    prediction = float(prediction[0][0])
    try:
        scaler = joblib.load('scaler.save')
        prediction = np.array([prediction]).reshape(-1, 1)
            

        dataset_sc = np.exp(scaler.inverse_transform(prediction))
        res = json.dumps({
            "prediction": int(round(dataset_sc.flatten()[0]))
        })
        return res, content_type
    except:
        raise ValueError('{{"error": "could not postprocess output data"}}')
    
    
def model_fn(model_dir):
    print("processing predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net(1, hidden_size, 1))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--num-gpus', type=int, default=0)
    
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args, _ = parser.parse_known_args()


    train(args)