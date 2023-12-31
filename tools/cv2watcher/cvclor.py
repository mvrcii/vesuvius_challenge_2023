import tkinter as tk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk, ImageDraw


# Function to handle click events
def on_click(event, canvas, image, draw, app):
    app.handle_click(event.x, event.y, canvas, draw)


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.click_count = 0
        # colors supported by tkinter:
        self.colors = ["red", "pink", "green"]
        self.background_path = "img1bw.png"
        self.foreground_path = "img2bw.png"
        self.background_window = None
        self.foreground_window = None
        self.setup_ui()

        # Create a frame to contain the buttons
        button_frame = tk.Frame(root)
        button_frame.pack()

        # Create four buttons and add them to the frame
        button1 = tk.Button(button_frame, text="Quartile 1", command=lambda: self.show_quartile(1))
        button2 = tk.Button(button_frame, text="Quartile 2", command=lambda: self.show_quartile(2))
        button3 = tk.Button(button_frame, text="Quartile 3", command=lambda: self.show_quartile(3))
        button4 = tk.Button(button_frame, text="Quartile 4", command=lambda: self.show_quartile(4))
        button5 = tk.Button(button_frame, text="Calculate Homography", command=self.calculate_homography)
        button1.pack()
        button2.pack()
        button3.pack()
        button4.pack()
        button5.pack()

        self.background_points = []
        self.foreground_points = []

        self.foreground_image = None
        self.background_image = None

        self.quartile_num = None
        self.quartiles = [None for _ in range(4)]

    def setup_ui(self):
        self.load_image(self.background_path, is_background=True)
        self.load_image(self.foreground_path, is_background=False)

    def load_image(self, file_path, is_background=False, quartile=0):

        window = tk.Toplevel(self.root)
        window.title("Background" if is_background else "Foreground")

        if is_background:
            self.reset_points()
            self.background_window = window
            window.geometry("800x800")
        else:
            self.foreground_window = window
            window.state('zoomed')

        canvas = tk.Canvas(window, cursor="cross")
        canvas.pack(fill="both", expand=True)

        pil_image = Image.open(file_path)
        if quartile == 1:
            pil_image = pil_image.crop((0, 0, pil_image.width // 2, pil_image.height // 2))
        elif quartile == 2:
            pil_image = pil_image.crop((pil_image.width // 2, 0, pil_image.width, pil_image.height // 2))
        elif quartile == 3:
            pil_image = pil_image.crop((0, pil_image.height // 2, pil_image.width // 2, pil_image.height))
        elif quartile == 4:
            pil_image = pil_image.crop((pil_image.width // 2, pil_image.height // 2, pil_image.width, pil_image.height))

        draw = ImageDraw.Draw(pil_image)
        tk_image = ImageTk.PhotoImage(pil_image)

        canvas.create_image(0, 0, anchor="nw", image=tk_image)
        canvas.bind("<Button-1>", lambda event: self.handle_click(event, canvas, draw))
        canvas.image = tk_image  # Keep a reference to avoid garbage collection
        canvas.pil_image = pil_image  # Store the PIL Image object

        if is_background:
            self.background_image = pil_image
        else:
            self.foreground_image = pil_image
        print("")

    def reset_points(self):
        self.click_count = 0
        self.background_points = []
        self.foreground_points = []

    def handle_click(self, event, canvas, draw):
        x = event.x
        y = event.y
        color = self.colors[self.click_count // 2 % len(self.colors)]
        radius = 2  # Size of the point
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=color)
        print(x, y)

        # Create a new PhotoImage from the modified PIL Image
        tk_image = ImageTk.PhotoImage(image=canvas.pil_image)
        canvas.create_image(0, 0, anchor="nw", image=tk_image)
        canvas.image = tk_image  # Update the canvas image reference

        pt = np.array([x, y])

        if self.click_count % 2 == 0:
            print("adding background point")
            self.background_points.append(pt)
        else:
            print("adding foreground point")
            self.foreground_points.append(pt)

        self.click_count += 1

    # Sample functions
    def show_quartile(self, num):
        self.quartile_num = num
        self.background_window.destroy()
        self.load_image(self.background_path, is_background=True, quartile=num)

    def calculate_homography(self):
        src_points = np.array(self.foreground_points)
        dst_points = np.array(self.background_points)

        print("src")
        print(src_points)
        print("dst")
        print(dst_points)
        M, _ = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)

        morphedForeground = cv.warpPerspective(cv.imread(self.foreground_path), M,
                                               (self.background_image.size[0], self.background_image.size[1]))
        # save the image
        print(M)
        cv.imwrite("morphed.png", morphedForeground)
        self.quartiles[self.quartile_num - 1] = morphedForeground
        for x in self.quartiles:
            print(type(x))
        all_there = True
        for x in self.quartiles:
            if x is None:
                all_there = False
        if all_there:
            self.combine_and_save()

    def combine_and_save(self):
        # combine the quartiles
        top = np.concatenate((self.quartiles[0], self.quartiles[1]), axis=1)
        bottom = np.concatenate((self.quartiles[2], self.quartiles[3]), axis=1)
        combined = np.concatenate((top, bottom), axis=0)
        cv.imwrite("combined.png", combined)



# Create the main window and an instance of the application
root = tk.Tk()
root.geometry("400x400")  # Change the width and height as desired

app = ImageApp(root)
root.mainloop()
