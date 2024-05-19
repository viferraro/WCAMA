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
diretorio_pai = criar_diretorio_incrementado('resultadosResNet', 'resNetMNIST')
print(f'Diretório criado: {diretorio_pai}')

# Cria o diretório 'ResNetCarbon'
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

# Função para treinar e validar um modelo
def treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, epocas):
    modelo.train()
    inicio_tempo = datetime.now()
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
        print(f'Época {epoca + 1}, Perda Treino: {perda_treino:.4f}, Precisão Treino: {acuracia_treino:.4f}')

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
        print(f'Época {epoca + 1}, Perda Validação: {perda_validacao:.4f}, Precisão Validação: {acuracia_validacao:.4f}')
    fim_tempo = datetime.now()
    tempo_treino = (fim_tempo - inicio_tempo)
    tempos_treino.append(tempo_treino.total_seconds())
    tracker.epoch_end()
    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia

# Treinamento e seleção do melhor modelo entre 10 candidatos
numero_modelos = 10
medias_perda_validacao = []
indice_melhor_modelo = -1
melhor_modelo = modelo
modelos = []
metricas = []
media_metricas = []
for i in range(numero_modelos):
    print("______________________________________________________________________________________________________")
    print(f'Treinando modelo {i + 1}/{numero_modelos}')
    entrada = torch.randn(1, 1, 28, 28).to(dispositivo)
    modelo = ResNet34().to(dispositivo)
    flops, parametros = profile(modelo, inputs=(entrada,), verbose=False)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)
    perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia = (
        treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, 20))
    metricas.append((perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino.total_seconds(), consumo_energia))
    # Calcular a média das métricas após o treino de cada modelo
    media_perda_treino = np.mean([m[0] for m in metricas])
    media_acuracia_treino = np.mean([m[1] for m in metricas])
    media_perda_validacao = np.mean([m[2] for m in metricas])
    media_acuracia_validacao = np.mean([m[3] for m in metricas])
    print(f'Modelo {i + 1}: Média Perda Treino: {media_perda_treino:.4f}, Média Acurácia Treino: {media_acuracia_treino:.4f}, '
          f'Média Perda Validação: {media_perda_validacao:.4f}, Média Acurácia Validação: {media_acuracia_validacao:.4f}')
    print(f'Tempo de treino: {tempo_treino}')
    print(f'FLOPs: {flops}')
    print(f'Parâmetros: {parametros}')
    print(f'Consumo de energia: {consumo_energia} W')
    medias_perda_validacao.append(media_perda_validacao)
    media_metricas.append(
        (media_perda_treino, media_acuracia_treino, media_perda_validacao, media_acuracia_validacao, tempo_treino.total_seconds(),
         consumo_energia))
    modelos.append(modelo)

# Cria um DataFrame com as métricas médias e salva em um arquivo Excel
df_metricas = pd.DataFrame(media_metricas, columns=['Média Perda Treino', 'Média Precisão Treino', 'Média Perda Validação',
                                                    'Média Precisão Validação', 'TempoTreino', 'ConsumoEnergia'])

# Adiciona uma coluna 'Modelo_x' ao DataFrame
nomes_modelos = ['Modelo_' + str(i + 1) for i in range(numero_modelos)]
df_metricas.insert(0, 'Modelo', nomes_modelos)

# Salva as métricas de todos os modelos em um único arquivo no diretório pai
df_metricas.to_excel(f'{diretorio_pai}/metricas_modelos.xlsx', index=False)

# Seleciona o melhor modelo com base na menor perda de validação
indice_melhor_modelo = medias_perda_validacao.index(min(medias_perda_validacao))

melhor_modelo = modelos[indice_melhor_modelo]
print('************************************************************************************************')
print(f'O melhor modelo é o {nomes_modelos[indice_melhor_modelo]} com a menor média de perda de validação: {media_perda_validacao:.4f}')
print('************************************************************************************************')

# Calcular a média dos tempos de treino e consumo de energia
media_tempo_treino = np.mean(tempos_treino)
media_consumo_energia = np.mean(potencias_treino)
print(f'Tempo Médio de Treino: {media_tempo_treino} segundos')
print(f'Consumo Médio de Energia: {media_consumo_energia} W')

# Inicializa listas para armazenar métricas de todas as inferências
acuracias = []
precisoes = []
revocacoes = []
pontuacoes_f1 = []
tempos_teste = []

# Realiza 10 inferências e armazena as métricas
for i in range(10):
    y_verdadeiros = []
    y_previstos = []
    inicio_tempo_teste = datetime.now()
    melhor_modelo.eval()
    with torch.no_grad():
        for dados in carregador_teste:
            imagens, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
            saidas = melhor_modelo(imagens)
            _, previstos = torch.max(saidas.data, 1)
            y_verdadeiros.extend(rotulos.cpu().numpy())
            y_previstos.extend(previstos.cpu().numpy())
    fim_tempo_teste = datetime.now()

    # Calcula as métricas para a inferência atual
    acuracias.append(accuracy_score(y_verdadeiros, y_previstos))
    precisoes.append(precision_score(y_verdadeiros, y_previstos, average='macro'))
    revocacoes.append(recall_score(y_verdadeiros, y_previstos, average='macro'))
    pontuacoes_f1.append(f1_score(y_verdadeiros, y_previstos, average='macro'))
    tempos_teste.append((fim_tempo_teste - inicio_tempo_teste).total_seconds())

# Calcula a média das métricas
media_acuracia = sum(acuracias) / len(acuracias)
media_precisao = sum(precisoes) / len(precisoes)
media_revocacao = sum(revocacoes) / len(revocacoes)
media_f1 = sum(pontuacoes_f1) / len(pontuacoes_f1)
media_tempo_teste = sum(tempos_teste) / len(tempos_teste)

# Imprime as médias das métricas
print(f'Média da Acurácia: {media_acuracia}')
print(f'Média da Precisão: {media_precisao}')
print(f'Média do Recall: {media_revocacao}')
print(f'Média do F1 Score: {media_f1}')
print(f'Média do Tempo de Teste: {media_tempo_teste} segundos')

sys.stdout.close()
sys.stdout = saida_padrao_original

# Calcular a matriz de confusão
cm = confusion_matrix(y_verdadeiros, y_previstos)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.xlabel('Previstos')
plt.ylabel('Verdadeiros')

# Salvar a figura
plt.savefig(f'{diretorio_pai}/matriz_confusao.png')
plt.close()

# Salva as médias das métricas em um arquivo
with open(f'{diretorio_pai}/metricas_medias_modelo.txt', 'w') as f:
    f.write(f'Média da Acurácia: {media_acuracia}\n')
    f.write(f'Média da Precisão: {media_precisao}\n')
    f.write(f'Média do Recall: {media_revocacao}\n')
    f.write(f'Média do F1 Score: {media_f1}\n')
    f.write(f'Média do Tempo de Teste: {media_tempo_teste} segundos\n')

pynvml.nvmlShutdown()
tracker.stop()
print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')