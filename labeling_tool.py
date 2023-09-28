import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
import os
import csv
from PIL import Image, ImageTk

class LabelingTool:
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]
        self.labeled_images = self.get_labeled_images()
        self.current_index = self.get_start_index()
        
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        
        self.entry = tk.Entry(root)
        self.entry.pack()
        self.entry.bind('<Return>', self.enter_key_pressed)
        
        self.next_button = tk.Button(root, text="Next & Save", command=self.next_image)
        self.next_button.pack()

        if self.current_index < len(self.image_files):
            self.load_image()
        else:
            self.root.quit()

    def get_labeled_images(self):
        if os.path.exists('labels.csv'):
            with open("labels.csv", "r", newline="") as f:
                reader = csv.reader(f)
                labeled = [row[0] for row in reader]
                return labeled
        return []

    def get_start_index(self):
        for idx, img in enumerate(self.image_files):
            if img not in self.labeled_images:
                return idx
        return len(self.image_files)

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
        image_array = np.load(image_path).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array)
        self.current_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
    def next_image(self):
        label = self.entry.get()
        
        with open("labels.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.image_files[self.current_index], label])
        
        # Clear the entry widget
        self.entry.delete(0, tk.END)
        
        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.load_image()
        else:
            self.root.quit()

    def enter_key_pressed(self, event):
        self.next_image()

root = tk.Tk()
root.title("Labeling Tool")
tool = LabelingTool(root, "C:\\mscode\\loa_utility\\imgsave\\lockoff_training")
root.mainloop()
