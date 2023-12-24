import json
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk, ImageOps


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        root.title("Image Editor")
        root.state('zoomed')  # Fullscreen

        # Canvas for images
        self.canvas = tk.Canvas(root, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Load images
        self.image1 = None
        self.image2 = None
        self.image1_id = None
        self.image2_id = None

        # Transformation parameters
        self.translate_x = 0
        self.translate_y = 0
        self.angle = 0
        self.scale_x = 1
        self.scale_y = 1

        # Buttons
        self.create_buttons()

    def load_config(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "r") as f:
                config = json.load(f)
                self.translate_x = config["translate_x"]
                self.translate_y = config["translate_y"]
                self.angle = config["angle"]
                self.scale_x = config["scale_x"]
                self.scale_y = config["scale_y"]
                self.display_images()

    def create_buttons(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.BOTTOM)

        tk.Button(frame, text='Load Image 1', command=self.load_image1).pack(side=tk.LEFT)
        tk.Button(frame, text='Load Image 2', command=self.load_image2).pack(side=tk.LEFT)
        tk.Button(frame, text='Save config', command=self.save_config).pack(side=tk.LEFT)
        tk.Button(frame, text='Load config', command=self.load_config).pack(side=tk.LEFT)
        tk.Button(frame, text='Save', command=self.combine_and_save).pack(side=tk.LEFT)
        tk.Button(frame, text='Load preset', command=self.load_preset).pack(side=tk.LEFT)

    def load_preset(self):
        self.load_image1(r"A:\projects_a\Python\kaggle1st\tools\img1.png")
        self.load_image2(r"A:\projects_a\Python\kaggle1st\tools\img2.png")
        # self.load_image1(r"C:\Users\Micha\Desktop\img1.png")
        # self.load_image2(r"C:\Users\Micha\Desktop\img2.png")

    def load_image1(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename()

        if file_path:
            self.image1 = Image.open(file_path)
            self.display_images()

    def load_image2(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename()
        if file_path:
            self.image2 = Image.open(file_path)
            self.display_images()

    def save_config(self):
        config = {"translate_x": self.translate_x, "translate_y": self.translate_y, "angle": self.angle,
                  "scale_x": self.scale_x, "scale_y": self.scale_y}
        with open("config.json", "w") as f:
            json.dump(config, f)


    def translate_image(self, dx, dy):
        self.translate_x += dx
        self.translate_y += dy
        self.display_images()

    def rotate_image(self, da):
        self.angle += da
        self.display_images()

    def scale_image(self, sx, sy):
        self.scale_x *= sx
        self.scale_y *= sy
        self.display_images()

    def combine_and_save(self):

        # Create a new blank image
        combined_image = Image.new('RGBA', (self.canvas.winfo_width(), self.canvas.winfo_height()))

        # Paste the first image onto the combined image
        if self.image1:
            img1 = self.image1
            img1_x = int(self.canvas.winfo_width() / 2 - img1.width / 2)
            img1_y = int(self.canvas.winfo_height() / 2 - img1.height / 2)

            # Check if img1 has an alpha channel
            if img1.mode == 'RGBA':
                combined_image.paste(img1, (img1_x, img1_y), img1)
            else:
                combined_image.paste(img1, (img1_x, img1_y))

        # Paste the second image onto the combined image
        if self.image2:
            img2 = ImageOps.exif_transpose(self.image2)
            img2 = img2.rotate(self.angle, expand=1, resample=Image.BICUBIC)
            img2 = img2.resize((int(img2.width * self.scale_x), int(img2.height * self.scale_y)), Image.ANTIALIAS)

            img2_x = int(self.canvas.winfo_width() / 2 + self.translate_x - img2.width / 2)
            img2_y = int(self.canvas.winfo_height() / 2 + self.translate_y - img2.height / 2)

            # Check if img2 has an alpha channel
            if img2.mode == 'RGBA':
                combined_image.paste(img2, (img2_x, img2_y), img2)
            else:
                combined_image.paste(img2, (img2_x, img2_y))

        # Save the combined image
        combined_image.save("combined_image.png")

    def display_images(self):

        # Display Image 1 (centered)
        if self.image1:
            img1 = ImageTk.PhotoImage(self.image1)
            if self.image1_id:
                self.canvas.itemconfig(self.image1_id, image=img1)
            else:
                self.image1_id = self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                                          image=img1, anchor='center')
            self.canvas.image1 = img1

        # Display Image 2 (manipulated)
        if self.image2:
            img2 = ImageOps.exif_transpose(self.image2)
            img2 = img2.rotate(self.angle, expand=1, resample=Image.BICUBIC)
            img2 = img2.resize((int(img2.width * self.scale_x), int(img2.height * self.scale_y)), Image.ANTIALIAS)
            img2_tk = ImageTk.PhotoImage(img2)
            if self.image2_id:
                self.canvas.itemconfig(self.image2_id, image=img2_tk)
                # Update position
                self.canvas.coords(self.image2_id, self.canvas.winfo_width() / 2 + self.translate_x,
                                   self.canvas.winfo_height() / 2 + self.translate_y)
            else:
                self.image2_id = self.canvas.create_image(self.canvas.winfo_width() / 2 + self.translate_x,
                                                          self.canvas.winfo_height() / 2 + self.translate_y,
                                                          image=img2_tk, anchor='center')
            self.canvas.image2 = img2_tk


def on_key_press(event):
    if event.char == 'w':
        app.translate_image(0, -3)
    elif event.char == 'a':
        app.translate_image(-3, 0)
    elif event.char == 's':
        app.translate_image(0, 3)
    elif event.char == 'd':
        app.translate_image(3, 0)
    elif event.char == 'q':
        app.rotate_image(-0.1)
    elif event.char == 'e':
        app.rotate_image(0.1)
    elif event.char == 'z':
        app.scale_image(0.99, 1)
    elif event.char == 'c':
        app.scale_image(1.01, 1)
    elif event.char == 'x':
        app.scale_image(1, 0.99)
    elif event.char == 'v':
        app.scale_image(1, 1.01)
    # same for uppercase letters with bigger magnitudes
    elif event.char == 'W':
        app.translate_image(0, -20)
    elif event.char == 'A':
        app.translate_image(-20, 0)
    elif event.char == 'S':
        app.translate_image(0, 20)
    elif event.char == 'D':
        app.translate_image(20, 0)
    elif event.char == 'Q':
        app.rotate_image(-5)
    elif event.char == 'E':
        app.rotate_image(5)
    elif event.char == 'Z':
        app.scale_image(0.9, 1)
    elif event.char == 'C':
        app.scale_image(1.1, 1)
    elif event.char == 'X':
        app.scale_image(1, 0.9)
    elif event.char == 'V':
        app.scale_image(1, 1.1)


root = tk.Tk()
root.bind('<Key>', on_key_press)
app = ImageEditorApp(root)
root.mainloop()
