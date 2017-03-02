import networkx as nx
import matplotlib.pyplot as plt
import networkx.drawing.nx_pydot as dot

G = nx.Graph()
nx.draw(dot.read_dot('C:/Users/vdthoang.2016/Google Drive/SMU COURSE/Software Mining and Analysis/Assignment 2/cfg_method.dot'))
plt.show()
