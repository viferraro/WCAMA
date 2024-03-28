import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
from datetime import datetime
from carbontracker.tracker import CarbonTracker
from thop import profile
from torchsummary import summary
import pynvml
import seaborn as sns
import matplotlib.pyplot as plt
import io
import sys

# Valor usado para inicializar o gerador de números aleatórios
SEED = 10

# Verificação da disponibilidade da GPU e seleção do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo utilizado: {device}')

for i in range(1):
    # Inicialização do NVML para monitoramento da GPU
    pynvml.nvmlInit()

    # Hiperparâmetros e inicializações
    max_epochs = 20
    tracker = CarbonTracker(epochs=max_epochs)

    # Carregamento e normalização do conjunto de dados CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Definição da arquitetura da rede neural ResNet50
    class ResNet50(nn.Module):
        def __init__(self):
            super(ResNet50, self).__init__()  # Inicializa a classe pai, nn.Module
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)  # Primeira camada convolucional
            self.bn1 = nn.BatchNorm2d(64)  # Normalização em lote para acelerar o treinamento
            self.relu = nn.ReLU(inplace=True)  # Função de ativação ReLU
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Camada de agrupamento máximo
            self.layer1 = self._make_layer(64, 64, 3)  # Primeiro bloco residual
            self.layer2 = self._make_layer(256, 128, 4, stride=2)  # Segundo bloco residual
            self.layer3 = self._make_layer(512, 256, 6, stride=2)  # Terceiro bloco residual
            self.layer4 = self._make_layer(1024, 512, 3, stride=2)  # Quarto bloco residual
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Camada de agrupamento médio adaptativo
            self.fc = nn.Linear(2048, 10)  # Camada totalmente conectada para classificação

        def _make_layer(self, inplanes, planes, blocks, stride=1):
            downsample = None
            # Verifica se é necessário uma camada de downsample
            if stride != 1 or inplanes != planes * 4:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride),
                    # Camada convolucional para downsample
                    nn.BatchNorm2d(planes * 4),  # Normalização em lote
                )

            layers = []
            layers.append(Bottleneck(inplanes, planes, stride,
                                     downsample))  # Adiciona o primeiro bloco com downsample se necessário
            inplanes = planes * 4
            for _ in range(1, blocks):
                layers.append(Bottleneck(inplanes, planes))  # Adiciona os blocos restantes

            return nn.Sequential(*layers)  # Retorna os blocos como uma sequência

        def forward(self, x):
            x = self.conv1(x)  # Passa a entrada pela primeira camada convolucional
            x = self.bn1(x)  # Normalização em lote
            x = self.relu(x)  # Função de ativação ReLU
            x = self.maxpool(x)  # Agrupamento máximo

            x = self.layer1(x)  # Passa a entrada pelo primeiro bloco residual
            x = self.layer2(x)  # Passa a entrada pelo segundo bloco residual
            x = self.layer3(x)  # Passa a entrada pelo terceiro bloco residual
            x = self.layer4(x)  # Passa a entrada pelo quarto bloco residual

            x = self.avgpool(x)  # Agrupamento médio adaptativo
            x = torch.flatten(x, 1)  # Achata o tensor para a camada totalmente conectada
            x = self.fc(x)  # Passa a entrada pela camada totalmente conectada

            return x  # Retorna a saída da rede


    class Bottleneck(nn.Module):
        expansion = 4  # Fator de expansão para a última camada convolucional no bloco

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(Bottleneck, self).__init__()  # Inicializa a classe pai, nn.Module
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)  # Primeira camada convolucional
            self.bn1 = nn.BatchNorm2d(planes)  # Normalização em lote
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1)  # Segunda camada convolucional
            self.bn2 = nn.BatchNorm2d(planes)  # Normalização em lote
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)  # Terceira camada convolucional
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # Normalização em lote
            self.relu = nn.ReLU(inplace=True)  # Função de ativação ReLU
            self.downsample = downsample  # Camada de downsample, se necessário
            self.stride = stride  # Passo para a segunda camada convolucional

        def forward(self, x):
            identity = x  # Salva a entrada original para a conexão residual

            out = self.conv1(x)  # Passa a entrada pela primeira camada convolucional
            out = self.bn1(out)  # Normalização em lote
            out = self.relu(out)  # Função de ativação ReLU

            out = self.conv2(out)  # Passa a entrada pela segunda camada convolucional
            out = self.bn2(out)  # Normalização em lote
            out = self.relu(out)  # Função de ativação ReLU

            out = self.conv3(out)  # Passa a entrada pela terceira camada convolucional
            out = self.bn3(out)  # Normalização em lote

            if self.downsample is not None:
                identity = self.downsample(x)  # Aplica o downsample na entrada original, se necessário

            out += identity  # Adiciona a entrada original (ou sua versão com downsample) à saída
            out = self.relu(out)  # Função de ativação ReLU

            return out  # Retorna a saída do bloco

    model = ResNet50().to(device)
    print(model)


    # Defina uma função para criar um diretório com incremento se ele já existir
    def create_incremented_dir(base_dir, subfolder_name):
        i = 1
        parent_dir = os.path.join(base_dir, f"{subfolder_name}_{i}")
        while os.path.exists(parent_dir):
            i += 1
            parent_dir = os.path.join(base_dir, f"{subfolder_name}_{i}")
        os.makedirs(parent_dir)
        return parent_dir


    # Crie o diretório pai 'leNet_x' com incremento se necessário
    parent_dir = create_incremented_dir('resultados', 'leNet')
    print(f'Diretório criado: {parent_dir}')

    # Salvar a saída padrão original
    original_stdout = sys.stdout

    # Redirecionar a saída padrão para um buffer de string
    sys.stdout = buffer = io.StringIO()

    summary(model, (3, 32, 32))

    # Obter o valor da string do buffer
    summary_str = buffer.getvalue()

    # Restaurar a saída padrão original
    sys.stdout = original_stdout

    # Salvar a string de resumo em um arquivo
    with open(f'{parent_dir}/model_summary.txt', 'w') as f:
        f.write(summary_str)

    # Função para treinar e validar um modelo
    def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs):
        model.train()
        for epoch in range(epochs):
            tracker.epoch_start()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total
            tracker.epoch_end()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

            # Validação
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        return train_loss, train_accuracy, val_loss, val_accuracy


    # Define uma função para criar um diretório se ele não existir
    def create_dir(base_dir, subfolder_name):
        base_dir = os.path.join(base_dir, subfolder_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir

    # Treinar 5 modelos e selecionar o melhor
    num_models = 5
    avg_valid_loss = []
    best_model_idx = -1
    best_model = model
    models = []
    metrics = []
    avg_metrics = []
    train_times = []
    train_powers = []

    # Cria o diretório pai 'resNet_x'
    parent_dir = create_dir('resultados', f'resNet_{i + 1}')

    for i in range(num_models):
        start_time = datetime.now()
        print(f'Training model {i + 1}/{num_models}')
        input = torch.randn(1, 3, 32, 32).to(device)
        model = ResNet50().to(device)
        flops, params = profile(model, inputs=(input,), verbose=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_loss, train_accuracy, val_loss, val_accuracy = (train_and_validate
                                                              (model, train_loader, val_loader,
                                                               criterion, optimizer, 20))
        end_time = datetime.now()
        train_time = (end_time - start_time)
        train_times.append(train_time.total_seconds())
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_usage = info / 1000.0
        train_powers.append(power_usage)
        metrics.append((train_loss, train_accuracy, val_loss, val_accuracy, train_time.total_seconds(), power_usage))

        # Calcular a média das métricas após o treino de cada modelo
        avg_train_loss = np.mean([m[0] for m in metrics])
        avg_train_accuracy = np.mean([m[1] for m in metrics])
        avg_val_loss = np.mean([m[2] for m in metrics])
        avg_val_accuracy = np.mean([m[3] for m in metrics])
        print(f'Model {i + 1}: Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}, '
              f'Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accuracy: {avg_val_accuracy:.4f}')
        print(f'Tempo de treino: {train_time}')
        print(f'FLOPs: {flops}')
        print(f'Parâmetros: {params}')
        print(f'Power usage: {power_usage} W')
        avg_valid_loss.append(avg_val_loss)
        avg_metrics.append(
            (avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy, train_time.total_seconds(), power_usage))
        models.append(model)

    # Crie um DataFrame com as métricas médias e salve-o em um arquivo Excel
    df_metrics = pd.DataFrame(avg_metrics, columns=['avg_Train Loss', 'avg_Train Accuracy', 'avg_Val Loss',
                                                    'avg_Val Accuracy', 'TrainTime', 'PowerUsage'])

    # Adicione uma coluna 'Model' ao DataFrame
    df_metrics.insert(0, 'Model', ['Modelo_' + str(i + 1) for i in range(num_models)])

    # Salve as métricas de todos os modelos em um único arquivo no diretório pai 'resNet_x'
    df_metrics.to_excel(f'{parent_dir}/models_metrics.xlsx', index=False)

    # Seleciona o melhor modelo com base na menor perda de validação.
    best_model_index = avg_valid_loss.index(min(avg_valid_loss))
    best_model = models[best_model_index]
    print('************************************************************************************************')
    print(
        f'O melhor modelo é o Modelo {best_model_index + 1} com a menor perda média de validação: {min(avg_valid_loss):.4f}')

    print('************************************************************************************************')
    # Calcular a média dos tempos de treino e power usage
    avg_train_time = np.mean(train_times)
    avg_power_usage = np.mean(train_powers)
    avg_metrics.append((avg_train_time, avg_power_usage))
    print(f'Average Train Time: {avg_train_time} seconds')
    print(f'Average Power Usage: {avg_power_usage} W')

    # Avaliar o melhor modelo no conjunto de teste
    y_true = []
    y_pred = []
    best_model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Imprimir métricas
    print(f'Accuracy: {accuracy}\n')
    print(f'Precision: {precision}\n')
    print(f'Recall: {recall}\n')
    print(f'F1 Score: {f1}\n')

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    # Salvar a figura
    plt.savefig(f'{parent_dir}/confusion_matrix.png')
    plt.close()

    #  Salvar as métricas do melhor modelo no diretório pai 'resNet_'
    with open(f'{parent_dir}/best_model_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')

    # salvar as saídas impressas em um arquivo
    with open(f'{parent_dir}/output.txt', 'w') as f:
        f.write(buffer.getvalue())

    pynvml.nvmlShutdown()
    tracker.stop()
    print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')