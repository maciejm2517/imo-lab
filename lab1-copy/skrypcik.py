import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv('kroa.csv',sep=' ')
positions=np.array([data['x'],data['y']]).T
print(positions)

fig, ax = plt.subplots(2, sharex=True, sharey=True)         # Prepare 2 plots
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter(positions[:, 0], positions[:, 1])             # plot A
ax[1].scatter(positions[:, 0], positions[:, 1])             # plot B
start_node = 0
distance = 0.
N=data.shape[0]
for i in range(N):
    start_pos = positions[start_node]
#     next_node = np.argmax(x_sol[start_node]) # needed because of MIP-approach used for TSP
#     end_pos = positions[next_node]
#     ax[1].annotate("",
#             xy=start_pos, xycoords='data',
#             xytext=end_pos, textcoords='data',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3"))
#     distance += np.linalg.norm(end_pos - start_pos)
#     start_node = next_node

textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14, # Textbox
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()