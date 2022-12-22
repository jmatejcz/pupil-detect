import tkinter
import time
import pandas as pd

root = tkinter.Tk()

# change to your screen resoulution
WIDTH = 1920
HEIGHT = 1080

# set window size
root.geometry(f"{WIDTH}x{HEIGHT}")
root['background'] = 'black'
root.attributes("-fullscreen", True)
canvas = tkinter.Canvas(root, width=WIDTH, height=HEIGHT, background='black')
canvas.pack()

table = pd.DataFrame()
timestamps = []
X = []
Y = []


class Calib():

    def __init__(self) -> None:
        self.oval_radius = 0
        self.oval_num = 0
        self.oval_centers = [
            (x, y) for y in [50, HEIGHT//2, HEIGHT-50] for x in [50, WIDTH//2, WIDTH-50]
        ]
        self.oval_center = self.oval_centers[self.oval_num]

    def increase_oval(self):
        canvas.create_oval(
            self.oval_center[0] - self.oval_radius,
            self.oval_center[1] - self.oval_radius,
            self.oval_center[0] + self.oval_radius,
            self.oval_center[1] + self.oval_radius,
            fill='green'
        )
        timestamps.append(time.time())
        X.append(self.oval_center[0])
        Y.append(self.oval_center[1])
        self.oval_radius += 1
        if self.oval_radius >= 25:
            self.decrease_oval()
        else:
            root.after(40, self.increase_oval)

    def decrease_oval(self):
        canvas.delete('all')
        canvas.create_oval(
            self.oval_center[0] - self.oval_radius,
            self.oval_center[1] - self.oval_radius,
            self.oval_center[0] + self.oval_radius,
            self.oval_center[1] + self.oval_radius,
            fill='green'
        )
        timestamps.append(time.time())
        X.append(self.oval_center[0])
        Y.append(self.oval_center[1])
        self.oval_radius -= 1
        if self.oval_radius <= 0:
            self.oval_num += 1
            if self.oval_num >= 9:
                root.destroy()
            else:
                self.oval_center = self.oval_centers[self.oval_num]
                self.increase_oval()

        else:
            root.after(40, self.decrease_oval)


calib = Calib()
root.after(0, calib.increase_oval)
root.mainloop()
data = {
    "x": X,
    "y": Y,
    "timestamp": timestamps
}
df = pd.DataFrame(data)
df.to_csv("callibration_data.csv")
