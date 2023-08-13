import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

style_image_path = None

def open_style_image():
    global style_image_path
    file_path = filedialog.askopenfilename()
    if file_path:
        style_image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        style_label.config(image=img)
        style_label.image = img
        stylize_button.config(state=tk.NORMAL)

def stylize_image():
    global style_image_path
    if style_image_path is None:
        return

    content_image_path = filedialog.askopenfilename()
    if content_image_path:
        hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
        
        content_image = tf.io.read_file(content_image_path)
        content_image = tf.image.decode_image(content_image, channels=3)
        content_image = tf.image.convert_image_dtype(content_image, tf.float32)
        content_image = content_image[tf.newaxis, :]
        
        style_image = tf.io.read_file(style_image_path)
        style_image = tf.image.decode_image(style_image, channels=3)
        style_image = tf.image.convert_image_dtype(style_image, tf.float32)
        style_image = style_image[tf.newaxis, :]
        
        stylized_image = hub_model(content_image, style_image)[0]
        stylized_image = tf.cast(tf.clip_by_value(stylized_image, 0, 1) * 255, tf.uint8)
        stylized_image = np.array(stylized_image)[0]
        
        stylized_img = Image.fromarray(stylized_image)
        stylized_img.save("stylized_image.jpg")
        save_label.config(text="Stylized image saved as stylized_image.jpg")

root = tk.Tk()
root.title("AI Image Stylization")

style_button = tk.Button(root, text="Open Style Image", command=open_style_image)
style_button.pack(pady=10)

style_label = tk.Label(root)
style_label.pack()

stylize_button = tk.Button(root, text="Apply Style", state=tk.DISABLED, command=stylize_image)
stylize_button.pack(pady=10)

save_label = tk.Label(root, text="")
save_label.pack()

root.mainloop()
