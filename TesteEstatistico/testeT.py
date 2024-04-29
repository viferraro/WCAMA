import numpy as np
from scipy import stats

# Define the carbon footprint and energy consumption values for each network
lenet5 = np.array([0.706, 47.071])
alexnet = np.array([3.374, 87.225])
resnet = np.array([2.721, 83.110])
mobilenet = np.array([0.807, 45.241])

# Group 1: LeNet-5 and MobileNet
group1 = np.array([lenet5, mobilenet])

# Group 2: AlexNet and ResNet
group2 = np.array([alexnet, resnet])

# Perform t-tests for carbon footprint and energy consumption
t_carbon, p_carbon = stats.ttest_ind(group1[:, 0], group2[:, 0])
t_energy, p_energy = stats.ttest_ind(group1[:, 1], group2[:, 1])

# Print the results
print(f"Carbon Footprint: t = {t_carbon}, p = {p_carbon}")
print(f"Energy Consumption: t = {t_energy}, p = {p_energy}")