import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from thop import profile
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from carbontracker.tracker import CarbonTracker
from carbontracker import parser
import pynvml

# Constante para inicialização do gerador de números aleatórios
SEED = 10

# Verifica se a GPU está disponível
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dispositivo)

# Inicializa o NVML para monitoramento da GPU
pynvml.nvmlInit()

# Função para criar um diretório com incremento, se necessário
def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Cria o diretório pai 'ResNetMNIST_' com incremento, se necessário
diretorio_pai = criar_diretorio_incrementado('resultados3', 'ResNetMNIST')
print(f'Diretório criado: {diretorio_pai}')

# Cria o diretório 'resNetCarbon'
diretorio_carbon = criar_diretorio_incrementado(diretorio_pai, 'resNet_carbono')
print(f'Diretório Carbono criado: {diretorio_carbon}')

# Definições iniciais
maximo_epocas = 20
tempos_treino = []
potencias_treino = []
tracker = CarbonTracker(epochs=maximo_epocas, monitor_epochs=-1, interpretable=True,
                           log_dir=f"./{diretorio_carbon}/",
                           log_file_prefix="cbt")

# Carrega e normaliza o MNIST
transformacao = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

conjunto_treino_completo = datasets.MNIST(root='./data', train=True, download=True, transform=transformacao)
conjunto_teste = datasets.MNIST(root='./data', train=False, download=True, transform=transformacao)

# Divide o conjunto de treino em treino e validação
tamanho_treino = int(0.8 * len(conjunto_treino_completo))
tamanho_validacao = len(conjunto_treino_completo) - tamanho_treino
conjunto_treino, conjunto_validacao = random_split(conjunto_treino_completo, [tamanho_treino, tamanho_validacao])

carregador_treino = DataLoader(conjunto_treino, batch_size=64, shuffle=True)
carregador_validacao = DataLoader(conjunto_validacao, batch_size=64, shuffle=False)
carregador_teste = DataLoader(conjunto_teste, batch_size=64, shuffle=False)

# Definição da arquitetura da rede neural ResNet34
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()  # Inicializa a classe pai, nn.Module
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)  # Primeira camada convolucional
        self.bn1 = nn.BatchNorm2d(64)  # Normalização em lote para acelerar o treinamento
        self.relu = nn.ReLU(inplace=True)  # Função de ativação ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Camada de agrupamento máximo
        self.layer1 = self._make_layer(64, 64, 3)  # Primeiro bloco residual
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # Segundo bloco residual
        self.layer3 = self._make_layer(128, 256, 6, stride=2)  # Terceiro bloco residual
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # Quarto bloco residual
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Camada de agrupamento médio adaptativo
        self.fc = nn.Linear(512, 10)  # Camada totalmente conectada para classificação

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        # Verifica se é necessário uma camada de downsample
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                # Camada convolucional para downsample
                nn.BatchNorm2d(planes),  # Normalização em lote
            )

        layers = []
        layers.append(BasicBlock(inplanes, planes, stride,
                                 downsample))  # Adiciona o primeiro bloco com downsample se necessário
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(inplanes, planes))  # Adiciona os blocos restantes

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


class BasicBlock(nn.Module):
    expansion = 1  # Fator de expansão para a última camada convolucional no bloco

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()  # Inicializa a classe pai, nn.Module
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1)  # Primeira camada convolucional
        self.bn1 = nn.BatchNorm2d(planes)  # Normalização em lote
        self.relu = nn.ReLU(inplace=True)  # Função de ativação ReLU
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1)  # Segunda camada convolucional
        self.bn2 = nn.BatchNorm2d(planes)  # Normalização em lote
        self.downsample = downsample  # Camada de downsample, se necessário
        self.stride = stride  # Passo para a primeira camada convolucional

    def forward(self, x):
        identity = x  # Salva a entrada original para a conexão residual

        out = self.conv1(x)  # Passa a entrada pela primeira camada convolucional
        out = self.bn1(out)  # Normalização em lote
        out = self.relu(out)  # Função de ativação ReLU

        out = self.conv2(out)  # Passa a entrada pela segunda camada convolucional
        out = self.bn2(out)  # Normalização em lote

        if self.downsample is not None:
            identity = self.downsample(x)  # Aplica o downsample na entrada original, se necessário

        out += identity  # Adiciona a entrada original (ou sua versão com downsample) à saída
        out = self.relu(out)  # Função de ativação ReLU

        return out  # Retorna a saída do bloco


modelo = ResNet34().to(dispositivo)
print(modelo)

# Salvar a saída padrão original
saida_padrao_original = sys.stdout

# Redirecionar a saída padrão para um buffer de string
sys.stdout = buffer = io.StringIO()

# Chamar a função summary
summary(modelo, (1, 28, 28))

# Obter o valor da string do buffer
resumo_str = buffer.getvalue()

# Restaurar a saída padrão original
sys.stdout = saida_padrao_original

# Salvar a string de resumo em um arquivo
with open(f'{diretorio_pai}/resumo_modelo.txt', 'w') as f:
    f.write(resumo_str)

# Salvar a saída padrão original
saida_padrao_original = sys.stdout

# Redirecionar a saída padrão para um arquivo
sys.stdout = open(f'{diretorio_pai}/saida.txt', 'w')

def treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, epocas):
    modelo.train()
    tempo_inicio = datetime.now()
    tracker.epoch_start()
    for epoca in range(epocas):
        tracker.epoch_start()
        perda_acumulada = 0.0
        corretos = 0
        total = 0
        for i, dados in enumerate(carregador_treino, 0):
            entradas, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = criterio(saidas, rotulos)
            perda.backward()
            otimizador.step()
            perda_acumulada += perda.item()
            _, previstos = torch.max(saidas.data, 1)
            total += rotulos.size(0)
            corretos += (previstos == rotulos).sum().item()

            # Medir o consumo de energia
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetPowerUsage(handle)
            consumo_energia = info / 1000.0
            potencias_treino.append(consumo_energia)

        perda_treino = perda_acumulada / len(carregador_treino)
        acuracia_treino = corretos / total
        print(f'Época {epoca + 1}, Perda Treino: {perda_treino:.4f}, Acurácia Treino: {acuracia_treino:.4f}')

        # Validação
        modelo.eval()
        perda_validacao = 0.0
        corretos = 0
        total = 0
        with torch.no_grad():
            for dados in carregador_validacao:
                imagens, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)
                perda_validacao += perda.item()
                _, previstos = torch.max(saidas.data, 1)
                total += rotulos.size(0)
                corretos += (previstos == rotulos).sum().item()
        tracker.epoch_end()
        perda_validacao /= len(carregador_validacao)
        acuracia_validacao = corretos / total
        print(f'Época {epoca + 1}, Perda Validação: {perda_validacao:.4f}, Acurácia Validação: {acuracia_validacao:.4f}')

    tempo_fim = datetime.now()
    tempo_treino = (tempo_fim - tempo_inicio)
    tempos_treino.append(tempo_treino.total_seconds())
    tracker.epoch_end()
    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino.total_seconds(), consumo_energia

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
        print(f'Training model {i+1}/{num_models}')
        input = torch.randn(1, 1, 28, 28).to(device)
        model = ResNet34().to(device)
        flops, params = profile(model, inputs=(input, ), verbose=False)
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

