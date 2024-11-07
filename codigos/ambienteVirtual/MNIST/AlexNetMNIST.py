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

# Cria o diretório pai 'alexNetMNIST_' com incremento, se necessário
diretorio_pai = criar_diretorio_incrementado('resultadosAlexNet', 'alexNetMNIST')
print(f'Diretório criado: {diretorio_pai}')

# Cria o diretório 'AlexNetCarbon'
diretorio_carbon = criar_diretorio_incrementado(diretorio_pai, 'alexNet_carbono')
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

carregador_treino = DataLoader(conjunto_treino, batch_size=32, shuffle=True)
carregador_validacao = DataLoader(conjunto_validacao, batch_size=32, shuffle=False)
carregador_teste = DataLoader(conjunto_teste, batch_size=32, shuffle=False)

# Definição da arquitetura da rede neural AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classificador = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # def forward(self, x):
    #
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classificador(x)
    #     return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #print(x.shape)  # Adicione esta linha
        x = torch.flatten(x, 1)
        #print(x.shape)  # Adicione esta linha
        x = self.classificador(x)
        return x

def inicializar_pesos(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

modelo = AlexNet().to(dispositivo)
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, 'min', patience=3, factor=0.1)
print(modelo)

modelo.apply(inicializar_pesos)

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
    tempo_inicio= datetime.now()
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

            # # Monitorar os gradientes
            # for nome, parametro in modelo.named_parameters():
            #     if parametro.grad is not None:
            #         print(f'Gradiente de {nome}: {parametro.grad.norm()}')

            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)  # Aplicar gradient clipping
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
        print(
            f'Época {epoca + 1}, Perda Validação: {perda_validacao:.4f}, Acurácia Validação: {acuracia_validacao:.4f}')
        scheduler.step(perda_validacao)

    tempo_fim = datetime.now()
    tempo_treino = (tempo_fim - tempo_inicio)
    tempos_treino.append(tempo_treino.total_seconds())
    tracker.epoch_end()
    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia


# Treinamento e seleção do melhor modelo entre 10 candidatos
numero_modelos = 10
medias_acuracia_validacao = []
indice_melhor_modelo = -1
melhor_modelo = None
modelos = []
metricas = []
media_metricas = []

for i in range(numero_modelos):
    print("______________________________________________________________________________________________________")
    print(f'Treinando modelo {i + 1}/{numero_modelos}')
    entrada = torch.randn(1, 1, 28, 28).to(dispositivo)
    #modelo = AlexNet().to(dispositivo)
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

    medias_acuracia_validacao.append(media_acuracia_validacao)
    media_metricas.append(
        (media_perda_treino, media_acuracia_treino, media_perda_validacao, media_acuracia_validacao,
         tempo_treino.total_seconds(),
         consumo_energia))
    modelos.append(modelo)

# Cria um DataFrame com as métricas médias e salva em um arquivo Excel
df_metricas = pd.DataFrame(media_metricas, columns=['Média Perda Treino', 'Média Acurácia Treino', 'Média Perda Validação',
                                                    'Média Acurácia Validação', 'TempoTreino', 'ConsumoEnergia'])

# Adiciona uma coluna 'Modelo_x' ao DataFrame
nomes_modelos = ['Modelo_' + str(i + 1) for i in range(numero_modelos)]
df_metricas.insert(0, 'Modelo', nomes_modelos)

# Salva as métricas de todos os modelos em um único arquivo no diretório pai
df_metricas.to_excel(f'{diretorio_pai}/metricas_modelos.xlsx', index=False)

# Seleciona o melhor modelo com base na maior acurácia de validação
indice_melhor_modelo = medias_acuracia_validacao.index(max(medias_acuracia_validacao))
melhor_modelo = modelos[indice_melhor_modelo]

print('************************************************************************************************')
print(f'O melhor modelo é o {nomes_modelos[indice_melhor_modelo]} com a maior média de acurácia de validação: {medias_acuracia_validacao[indice_melhor_modelo]:.4f}')
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
    precisoes.append(precision_score(y_verdadeiros, y_previstos, average='macro', zero_division=0))
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