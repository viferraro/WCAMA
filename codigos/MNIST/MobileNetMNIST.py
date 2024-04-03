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
from carbontracker import parser
from thop import profile
from torchsummary import summary
import pynvml
import seaborn as sns
import matplotlib.pyplot as plt
import io
import sys

# Valor usado para inicializar o gerador de números aleatórios
SEED = 10

# Verificar se a GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for i in range(1):
    # Inicialização do NVML para monitoramento da GPU
    pynvml.nvmlInit()


    # Defina uma função para criar um diretório com incremento, se ele já existir
    def create_incremented_dir(base_dir, subfolder_name):
        i = 1
        parent_dir = os.path.join(base_dir, f"{subfolder_name}_{i}")
        while os.path.exists(parent_dir):
            i += 1
            parent_dir = os.path.join(base_dir, f"{subfolder_name}_{i}")
        os.makedirs(parent_dir)
        return parent_dir


    # Crie o diretório pai 'alexNetMNIST_' com incremento se necessário
    parent_dir = create_incremented_dir('resultados15', 'MobileNetMNIST')
    print(f'Diretório criado: {parent_dir}')

    # Crie o diretório 'AlexNetCarbon'

    carbon_dir = create_incremented_dir(parent_dir, 'mobilenet_carbon')
    print(f'Diretório Carbon criado: {carbon_dir}')

    # Definições iniciais
    max_epochs = 20
    train_times = []
    train_powers = []
    tracker = CarbonTracker(epochs=max_epochs, monitor_epochs=-1, interpretable=True,
                            log_dir=f"./{carbon_dir}/",
                            log_file_prefix="cbt")

    log_dir = f"./{carbon_dir}/"
    all_logs = os.listdir(log_dir)
    std_logs = [f for f in all_logs if f.endswith('_carbontracker.log')]
    missing_logs = ['epoch_{}_carbontracker.log'.format(i) for i in range(max_epochs) if
                    'epoch_{}_carbontracker.log'.format(i) not in all_logs]
    for f in missing_logs:
        log_file = f + "_carbontracker.log"
        if log_file in std_logs:
            std_logs.remove(log_file)

    # Agora você pode chamar as funções do parser com segurança
    parser.print_aggregate(log_dir)
    logs = parser.parse_all_logs(log_dir)
    first_log = logs[0]

    print(f"Output file name: {first_log['output_filename']}")
    print(f"Standard file name: {first_log['standard_filename']}")
    print(f"Stopped early: {first_log['early_stop']}")
    print(f"Measured consumption: {first_log['actual']}")
    print(f"Predicted consumption: {first_log['pred']}")
    print(f"Measured GPU devices: {first_log['components']['gpu']['devices']}")

    # Carregar e normalizar o MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Dividir o conjunto de treino em treino e validação
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Definindo blocos de construção da MobileNet
    def conv_dw(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    # Arquitetura simplificada da MobileNet para MNIST
    class MobileNet(nn.Module):
        def __init__(self):
            super(MobileNet, self).__init__()
            self.model = nn.Sequential(
                conv_dw(1, 32, 1),  # MNIST é um canal (escala de cinza), 32 filtros
                nn.AvgPool2d(2, 2),
                conv_dw(32, 64, 1),
                nn.AvgPool2d(2, 2),
                conv_dw(64, 128, 1),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.25),
                nn.Flatten(),
                # Calcula o tamanho correto para a entrada da camada linear
                nn.Linear(128 * 3 * 3, 10)  # 10 classes
            )

        def forward(self, x):
            return self.model(x)


    # Criando o modelo
    model = MobileNet().to(device)
    print(model)

    # Salvar a saída padrão original
    original_stdout = sys.stdout

    # Redirecionar a saída padrão para um buffer de string
    sys.stdout = buffer = io.StringIO()

    # Chamar a função summary
    summary(model, (1, 28, 28))

    # Obter o valor da string do buffer
    summary_str = buffer.getvalue()

    # Restaurar a saída padrão original
    sys.stdout = original_stdout

    # Salvar a string de resumo em um arquivo
    with open(f'{parent_dir}/model_summary.txt', 'w') as f:
        f.write(summary_str)

    # Salvar a saída padrão original novamente
    original_stdout = sys.stdout

    # Redirecionar a saída padrão para um arquivo
    sys.stdout = open(f'{parent_dir}/output.txt', 'w')


    # Função para treinar e validar um modelo
    def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs):
        model.train()
        start_time = datetime.now()
        tracker.epoch_start()
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
                # Medir o consumo de energia
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_usage = info / 1000.0
                train_powers.append(power_usage)
            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

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
            tracker.epoch_end()
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        end_time = datetime.now()
        train_time = (end_time - start_time)
        train_times.append(train_time.total_seconds())
        tracker.epoch_end()
        return train_loss, train_accuracy, val_loss, val_accuracy, train_time, power_usage


    # Treinamento e seleção do melhor modelo entre 10 candidatos
    num_models = 10
    avg_valid_loss = []
    best_model_idx = -1
    best_model = model
    models = []
    metrics = []
    avg_metrics = []
    for i in range(num_models):
        print("______________________________________________________________________________________________________")
        print(f'Training model {i + 1}/{num_models}')
        input = torch.randn(1, 1, 28, 28).to(device)
        model = MobileNet().to(device)
        flops, params = profile(model, inputs=(input,), verbose=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_loss, train_accuracy, val_loss, val_accuracy, train_time, power_usage = (
            train_and_validate(model, train_loader, val_loader, criterion, optimizer, 20))
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
            (avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy, train_time.total_seconds(),
             power_usage))
        models.append(model)

    # Crie um DataFrame com as métricas médias e salve-o em um arquivo Excel
    df_metrics = pd.DataFrame(avg_metrics, columns=['avg_Train Loss', 'avg_Train Accuracy', 'avg_Val Loss',
                                                    'avg_Val Accuracy', 'TrainTime', 'PowerUsage'])

    # Adiciona uma coluna 'Modelo_x' ao DataFrame
    modelos = ['Modelo_' + str(i + 1) for i in range(num_models)]
    df_metrics.insert(0, 'Model', modelos)

    # Salva as métricas de todos os modelos em um único arquivo no diretório pai 'leNet_x'
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
    # avg_metrics.append((avg_train_time, avg_power_usage))
    print(f'Average Train Time: {avg_train_time} seconds')
    print(f'Average Power Usage: {avg_power_usage} W')

    # # Avaliar o melhor modelo no conjunto de teste
    # y_true = []
    # y_pred = []
    # start_time_test = datetime.now()
    # best_model.eval()
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = best_model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         y_true.extend(labels.cpu().numpy())
    #         y_pred.extend(predicted.cpu().numpy())
    # end_time_test = datetime.now()

    # # Calcular métricas
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average='macro')
    # recall = recall_score(y_true, y_pred, average='macro')
    # f1 = f1_score(y_true, y_pred, average='macro')
    # test_time = (end_time_test - start_time_test)
    # # Imprimir métricas
    # print(f'Accuracy: {accuracy}\n')
    # print(f'Precision: {precision}\n')
    # print(f'Recall: {recall}\n')
    # print(f'F1 Score: {f1}\n')
    # print(f'Test Time: {test_time}')
    # print(f'seconds: {test_time.total_seconds()}')

    # Inicializa listas para armazenar métricas de todas as inferências
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    test_times = []

    # Realiza 10 inferências e armazena as métricas
    for i in range(10):
        y_true = []
        y_pred = []
        start_time_test = datetime.now()
        best_model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = best_model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        end_time_test = datetime.now()

        # Calcula as métricas para a inferência atual
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, average='macro'))
        recalls.append(recall_score(y_true, y_pred, average='macro'))
        f1_scores.append(f1_score(y_true, y_pred, average='macro'))
        test_times.append((end_time_test - start_time_test).total_seconds())

    # Calcula a média das métricas
    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    mean_test_time = sum(test_times) / len(test_times)

    # Imprime as médias das métricas
    print(f'Média da Acurácia: {mean_accuracy}\n')
    print(f'Média da Precisão: {mean_precision}\n')
    print(f'Média do Recall: {mean_recall}\n')
    print(f'Média do F1 Score: {mean_f1}\n')
    print(f'Média do Tempo de Teste: {mean_test_time} segundos\n')

    sys.stdout.close()
    sys.stdout = original_stdout

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

    # #  Salvar as métricas do melhor modelo no diretório pai 'leNet_'
    # with open(f'{parent_dir}/best_model_metrics.txt', 'w') as f:
    #     f.write(f'Accuracy: {accuracy}\n')
    #     f.write(f'Precision: {precision}\n')
    #     f.write(f'Recall: {recall}\n')
    #     f.write(f'F1 Score: {f1}\n')
    #     f.write(f'Test Time: {test_time}\n')
    #     f.write(f'Seconds: {test_time.total_seconds()}\n')

    # Salva as médias das métricas em um arquivo
    with open(f'{parent_dir}/average_model_metrics.txt', 'w') as f:
        f.write(f'Média da Acurácia: {mean_accuracy}\n')
        f.write(f'Média da Precisão: {mean_precision}\n')
        f.write(f'Média do Recall: {mean_recall}\n')
        f.write(f'Média do F1 Score: {mean_f1}\n')
        f.write(f'Média do Tempo de Teste: {mean_test_time} segundos\n')

    pynvml.nvmlShutdown()
    tracker.stop()
    print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
