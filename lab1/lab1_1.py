import tkinter as tk
import numpy as np

def create_transformation_matrix_2d(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def create_scale_matrix_2d(sx, sy, sz=0):
    if sz == 0: return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

class Transformation2d:
    def __init__(self, root):
        self.root = root
        self.root.title("2D")
        self.size = 500

        self.m = np.array([
            [50, 0, 1],
            [0, 100, 1],
            [150, 100, 1],
            [200, 0, 1]
        ])
        self.transformed_m = self.m.copy()
        self.dx = 100
        self.dy = 100
        self.dx_sign = -1
        self.dy_sign = -1
        self.s = 1.0
        self.s_sign = 1

        self.canvas = tk.Canvas(self.root, width=500, height=500, bg='white')
        self.canvas.pack()
        self.animate()

    def animate(self):
        self.canvas.delete("all")

        # зміна координат
        if self.dx >= self.size - self.s * 100 or self.dx <= 0:
            self.dx_sign *= -1
        if self.dy >= self.size - self.s * 100 or self.dy <= 0:
            self.dy_sign *= -1
        self.dx += 5 * self.dx_sign
        self.dy += 5 * self.dy_sign

        # масштабування
        if self.s >= 2.0 or self.s <= 0.5:
            self.s_sign *= -1
        self.s += 0.05 * self.s_sign

        # draw figure
        scale_m = create_scale_matrix_2d(self.s, self.s)
        translate_m = create_transformation_matrix_2d(self.dx, self.dy)
        self.transformed_m = self.m @ scale_m.T @ translate_m.T
        points = self.transformed_m[:, :2].tolist()

        # оновлюємо канвас
        self.canvas.create_polygon(points, outline='blue', fill='lightblue')
        self.root.after(50, self.animate)

root = tk.Tk()
Transformation2d(root)
root.mainloop()