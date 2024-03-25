import torch
import torch.nn as nn
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

# Valor usado para inicializar o gerador de números aleatórios
SEED = 10

# Verificar se a GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for i in range(5):
    # Inicialização do NVML para monitoramento da GPU
    pynvml.nvmlInit()

    # Definições iniciais
    max_epochs = 20
    tracker = CarbonTracker(epochs=max_epochs)
    train_times = []
    power_usages = []

    # Carregar e normalizar o CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Dividir o conjunto de treino em treino e validação
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Definir a rede neural LeNet-5
    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1   = nn.Linear(16*5*5, 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, 10)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.batchnorm1 = nn.BatchNorm2d(6)
            self.batchnorm2 = nn.BatchNorm2d(16)

        def forward(self, x):
            x = self.pool(torch.relu(self.batchnorm1(self.conv1(x))))
            x = self.pool(torch.relu(self.batchnorm2(self.conv2(x))))
            x = x.view(-1, 16*5*5)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Função para treinar e validar um modelo
    def train_and_validate(model, train_loader, val_loader, criterion, optimizer, max_epochs):
        model.train()
        for epoch in range(max_epochs):
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

    # Treinar 10 modelos e selecionar o melhor
    num_models = 10
    best_val_loss = float('inf')
    best_model_idx = -1
    metrics = []
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    summary(model, (3,32,32))

    for i in range(num_models):
        timeStart = datetime.now()
        print(f'Training model {i+1}/{num_models}')
        input = torch.randn(1, 3, 32, 32).to(device)
        model = LeNet5().to(device)
        flops, params = profile(model, inputs=(input, ), verbose=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_loss, train_accuracy, val_loss, val_accuracy = train_and_validate(model, train_loader, val_loader, criterion, optimizer, 20)
        metrics.append((train_loss, train_accuracy, val_loss, val_accuracy))
        # Calcular a média das métricas após o treino de cada modelo
        avg_train_loss = np.mean([m[0] for m in metrics])
        avg_train_accuracy = np.mean([m[1] for m in metrics])
        avg_val_loss = np.mean([m[2] for m in metrics])
        avg_val_accuracy = np.mean([m[3] for m in metrics])
        timeEnd = datetime.now()
        tempoTreino = timeEnd - timeStart
        train_times.append(tempoTreino.total_seconds())  # Nova linha
        print(f'Model {i + 1}: Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accuracy: {avg_val_accuracy:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_idx = i
            best_model = model
        print(f'Tempo de treino: {tempoTreino}')
        print(f'FLOPs: {flops}')
        print(f'Parâmetros: {params}')
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_usage = info / 1000.0  # Nova linha
        power_usages.append(power_usage)  # Nova linha
        print(f'Power usage: {power_usage} W')

    # # Redirecione a saída padrão de volta ao normal
    # sys.stdout = old_stdout
    # # Agora a saída do CarbonTracker está na variável 'output'
    # carbontracker_output = output.getvalue()

    # Imprimir qual modelo foi escolhido como o melhor
    print(f'O melhor modelo é o Modelo {best_model_idx+1} com a menor perda média de validação: {best_val_loss:.4f}')

    # Calcular a média dos tempos de treino e power usage
    avg_train_time = np.mean(train_times)  # Nova linha
    avg_power_usage = np.mean(power_usages)  # Nova linha
    print(f'Average Train Time: {avg_train_time} seconds')  # Nova linha
    print(f'Average Power Usage: {avg_power_usage} W')  # Nova linha

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


    pynvml.nvmlShutdown()

    # verifica as pastas existentes
    def create_dir(base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        dirs = [d for d in dirs if 'teste_' in d]
        if dirs:
            max_index = max([int(d.split('_')[1]) for d in dirs])
            new_dir = os.path.join(base_dir, f'teste_{max_index + 1}')
        else:
            new_dir = os.path.join(base_dir, 'teste_1')
        os.makedirs(new_dir)
        return new_dir

    # Use a função para criar um novo diretório
    new_dir = create_dir('resultados')

    #  Imprimir e salvar a matriz de confusão
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.savefig(f'{new_dir}/confusion_matrix.png')

    # Salvar os resultados do CarbonTracker
    tracker_results = tracker.get_data()
    df_tracker_results = pd.DataFrame(tracker_results)
    df_tracker_results.to_csv(f'{new_dir}/carbontracker_results.csv', index=False)

    # Salvar métricas em um arquivo Excel
    df_metrics = pd.DataFrame(metrics, columns=['Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
    df_metrics['Train Time'] = train_times  # Nova linha
    df_metrics['Power Usage'] = power_usages  # Nova linha
    df_metrics.to_excel(f'{new_dir}/model_metrics.xlsx', index=False)

    # Salvar métricas do melhor modelo em um arquivo de texto
    with open(f'{new_dir}/best_model_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Train Time: {train_times[best_model_idx]} seconds\n')  # Nova linha
        f.write(f'Power Usage: {power_usages[best_model_idx]} W\n')  # Nova linha

    tracker.stop()
    print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')