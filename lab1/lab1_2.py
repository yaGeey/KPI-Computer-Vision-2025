import tkinter as tk
import numpy as np
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def project_iso(x, y, z):
    angle = 30 * math.pi / 180  # 30 градусів
    x2d = x - z * math.cos(angle)
    y2d = y - z * math.sin(angle)
    return x2d, y2d

class Transformation3d:
    def __init__(self, root):
        self.root = root
        self.root.title("3D")
        self.t = 0

        s = 100
        self.m = np.array([
            [0, 0, 0],
            [s, 0, 0],
            [s, 0, s],
            [0, 0, s],
            [s/2, s*2, s/2]
        ])
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 4), (2, 4), (3, 4)
        ]

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        self.animate()

    def animate(self):
        self.ax.clear()
        self.ax.axis('off')

        # колір з прозорістю
        r = (np.sin(self.t) + 1)/2
        g = (np.cos(self.t) + 1)/2
        b = 0.5
        a = (np.sin(self.t/2)+1)/2
        color = (r, g, b, a)

        # draw edges
        for i, j in self.edges:
            x1, y1 = project_iso(*self.m[i])
            x2, y2 = project_iso(*self.m[j])
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

        self.canvas.draw()
        self.t += 0.1
        self.root.after(50, self.animate)

root = tk.Tk()
Transformation3d(root)
root.mainloop()
