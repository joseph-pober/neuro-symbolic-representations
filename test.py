import matplotlib.pyplot as plt
import numpy as np

n = 2
m = n - 1

# Create a figure with 2x2 subplots
# fig, axes = plt.subfigs#plt.subplots(n, n)
topfig = plt.figure()
figs = topfig.subfigures(n, n)

topfig.tight_layout()
# Iterate over the subplots in the 2x2 grid
for i in range(n):
    for j in range(n):
        fig =figs[i, j]
        # For each subplot, create a 2x2 grid of smaller plots
        # ax = axes[i, j]
        
        # Create a smaller figure inside each subplot with 2x2 grid
        small_axes = fig.subplots(n,n)  # np.array([[None, None], [None, None]])  # 2x2 grid of subfigures
        
        for k in range(n):
            for l in range(n):
                # small_figs[k, j] = ax.inset_axes([n * 0.45, 1 - (m + 1) * 0.45, 0.4, 0.4])  # Define inset axes
                # small_fig = small_figs[k, l]
                ax = small_axes[k, l]
                
                x = np.linspace(0, 10, 100)
                y = np.sin(x + (i + j))  # Add different offset for variety
                
                # Plot the data
                ax.plot(x, y)
                # plt.imshow(img)
                # plt.gray()
                str1 = f"1"
                ax.set_title(str1 + str1 + str1, fontsize=8)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
                # small_fig.plot(x, y)
                # small_fig.set_title(f"Subplot {i * 2 + j + 1} ({m + 1},{n + 1})")
        
        # ax.set_title(f"Grid {i * 2 + j + 1} (2x2)")
        # ax.axis('off')  # Turn off the axis for the main grid plot

# Adjust layout
# plt.tight_layout()
plt.show()