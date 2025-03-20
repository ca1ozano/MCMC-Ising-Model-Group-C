import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data using pandas
df = pd.read_csv("magnetization_data.txt", delimiter=",")  # Adjust delimiter if necessary

# Convert to NumPy array if needed
data = df.to_numpy()

# Extract columns
T = data[:, 0]  # Temperature
B = data[:, 1]  # External magnetic field
M = data[:, 2]  # Magnetization

# Reshape data into a 2D grid
T_unique = np.unique(T)
B_unique = np.unique(B)
M_grid = M.reshape(len(B_unique), len(T_unique))  # Assuming data is ordered properly
print(M_grid)
# Create heatmap
plt.figure(figsize=(8, 6))
plt.imshow(M_grid.T, aspect='auto', origin='lower', 
           extent=[T_unique.min(), T_unique.max(), B_unique.min(), B_unique.max()], 
           cmap="coolwarm")  # Choose a colormap like "coolwarm" or "RdBu"

# Add labels and colorbar
plt.axvline(x=2.269, linestyle='--', color='black', label=r'$T_c = 2.269$')
plt.legend()
plt.colorbar(label="Magnetization")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetic Field (B)")
plt.title("Magnetization Heatmap")

plt.show()
