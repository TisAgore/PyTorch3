# -*- coding: utf-8 -*-
"""homework_regularization_experiments.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13KIBUk8u6Zy39h3ZMtw-MRBG57mGWtTt
"""

import torch

# Создадим даталоадеры на основе датасетов MNIST
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# класс FullyConnectedModel
import torch.nn as nn
import json


class FullyConnectedModel(nn.Module):
    def __init__(self, config_path=None, input_size=None, num_classes=None, **kwargs):
        super().__init__()

        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs

        self.input_size = input_size or self.config.get('input_size', 784)
        self.num_classes = num_classes or self.config.get('num_classes', 10)

        self.name: str = None
        self.layers = self._build_layers()

    def load_config(self, config_path):
        """Загружает конфигурацию из JSON файла"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _build_layers(self):
        """Строит слои модели, полученные из загруженной ранее конфигурации"""
        layers = []
        prev_size = self.input_size

        layer_config = self.config.get('layers', [])
        layers_amount = len(layer_config)
        if layers_amount <= 1:
          layers_prefix = 'layer'
        else:
          layers_prefix = 'layers'
        self.name = f'{layers_amount} {layers_prefix}'

        for layer_spec in layer_config:
            layer_type = layer_spec['type']

            if layer_type == 'linear':
                out_size = layer_spec['size']
                layers.append(nn.Linear(prev_size, out_size))
                prev_size = out_size

            elif layer_type == 'relu':
                layers.append(nn.ReLU())

            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())

            elif layer_type == 'tanh':
                layers.append(nn.Tanh())

            elif layer_type == 'dropout':
                rate = layer_spec.get('rate', 0.5)
                layers.append(nn.Dropout(rate))

            elif layer_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(prev_size))

            elif layer_type == 'layer_norm':
                layers.append(nn.LayerNorm(prev_size))

        layers.append(nn.Linear(prev_size, self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


def create_model_from_config(config_path, input_size=None, num_classes=None):
    """Создает модель из JSON конфигурации"""
    return FullyConnectedModel(config_path, input_size, num_classes)

# trainer
import torch.optim as optim
from tqdm import tqdm


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False, l2_alpha=0.001):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

# utils
import torch
import matplotlib.pyplot as plt
import os

def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_test_loss: float,
        best_test_acc: float
    ):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_test_loss': best_test_loss,
        'best_test_acc': best_test_acc
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def load_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['epoch'], state_dict['best_test_loss'], state_dict['best_test_acc']

"""# Задание 3: Эксперименты с регуляризацией (25 баллов)

## 3.1 Сравнение техник регуляризации (15 баллов)
"""

# Исследуйте различные техники регуляризации:
# Для каждого варианта:
# - Используйте одинаковую архитектуру

# - Без регуляризации
usual = {
    "input_size": 977,
    "num_classes": 10,
    "layers": [{"type": "linear", "size": 256},
               {'type': 'relu'},
               {'type': 'linear', 'size': 128},
               {'type': 'relu'},
               {'type':' linear', 'size': 64}]}

usual_model = FullyConnectedModel(**usual)

# - Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
only_dropout = {
    "input_size": 977,
    "num_classes": 10,
    "layers": [{"type": "linear", "size": 256},
               {'type': 'dropout', 'rate': 0.1},
               {'type': 'relu'},
               {'type': 'linear', 'size': 128},
               {'type': 'dropout', 'rate': 0.3},
               {'type': 'relu'},
               {'type':' linear', 'size': 64},
               {'type': 'dropout', 'rate': 0.5}]}
dropout_model = FullyConnectedModel(**only_dropout)

# - Только BatchNorm
only_batchNorm = {
    "input_size": 977,
    "num_classes": 10,
    "layers": [{"type": "linear", "size": 256},
               {'type': 'batch_norm'},
               {'type': 'relu'},
               {'type': 'linear', 'size': 128},
               {'type': 'batch_norm'},
               {'type': 'relu'},
               {'type':' linear', 'size': 64},
               {'type': 'batch_norm'}]}
batchNorm_model = FullyConnectedModel(**only_batchNorm)

# - Dropout + BatchNorm
dropout_batchNorm =  {
    "input_size": 977,
    "num_classes": 10,
    "layers": [{"type": "linear", "size": 256},
               {'type': 'batch_norm'},
               {'type': 'dropout', 'rate': 0.1},
               {'type': 'relu'},
               {'type': 'linear', 'size': 128},
               {'type': 'batch_norm'},
               {'type': 'dropout', 'rate': 0.3},
               {'type': 'relu'},
               {'type':' linear', 'size': 64},
               {'type': 'batch_norm'},
               {'type': 'dropout', 'rate': 0.5}]}
dropbatch_model = FullyConnectedModel(**dropout_batchNorm)

search_models = {
    'usual' : usual_model,
    'dropout': dropout_model,
    'dropout_batchnorm': dropbatch_model,
    'batchNorm' : batchNorm_model}

device = torch.device('cuda')
train_loader, test_loader = get_mnist_loaders()
search_loss = {}
for model_name in search_models.keys():
    model = search_models[model_name]
    model = model.to(device)
    search_loss[model_name] = train_model(model, train_loader, test_loader, epochs=10, device=device)
    print(search_loss[model_name])

#### Сравните финальную точность
fig, ax = plt.subplots(2,2, figsize=(16, 9))

for k, values in search_loss.items():
    for i, val in enumerate(values.values()):
        nrow = i // 2
        ncol = i % 2
        ax[nrow, ncol].plot(val, label=k)
        ax[nrow, ncol].legend()
        ax[nrow, ncol].grid(True)

ax[0, 0].set_title('Train Losses')
ax[0, 1].set_title('Train Accuracy')
ax[1, 0].set_title('Test Losses')
ax[1, 1].set_title('Test Accuracy')

plt.show()

"""- Проанализируйте стабильность обучения
> Как мы видим, самые низкие потери и высокие значения на тренировочных данных (причем ведет себя очень стабильно( - usual layer. Она же дает самые высокие потери и низкую точность относительно других моделей на тестовой ввыборке (вела себя нестабильно)4
>
> Dropout дает высокие и нестабильные потери, но стабильную точность
>
> Самыми лучшими и стабильными относительно других на тестовой выборке были **Dropout + BatchNorm** и обычный **BatchNorm**. **Dropout + BatchNorm** был самым худшим по потерям и accuracy на train выборке, но вырвался вперед на test выборке, показав практически ***такой же результат***, как и на train выборке

"""