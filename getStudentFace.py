import cv2 as cv
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime

class WebcamApp:
    def __init__(self, root, capture_directory, image_filename):
        self.root = root
        self.root.title("Webcam Capture App")

        self.capture_directory = capture_directory
        self.image_filename = image_filename

        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.capture_button = ttk.Button(root, text="Take Photo", command=self.capture_image)
        self.capture_button.pack(pady=10)

        # Open a connection to the webcam (usually 0 for the default camera)
        self.cap = cv.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Start the video streaming
        self.show_video_stream()

    def show_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            self.video_label.after(10, self.show_video_stream)

    def capture_image(self):
        # Capture a single frame
        ret, frame = self.cap.read()

        # Save the captured frame as an image
        image_path = os.path.join(self.capture_directory, self.image_filename)
        cv.imwrite(image_path, frame)

        print(f"Image captured and saved to {image_path}")

        # Release the webcam and close the Tkinter window
        self.cap.release()
        self.root.destroy()

student_name = input("Enter your name: ")

capture_directory = "Images"
image_filename = f"{student_name}.jpg"

# Create the main Tkinter window
root = tk.Tk()

# Create the WebcamApp instance
app = WebcamApp(root, capture_directory, image_filename)

# Start the Tkinter event loop
root.mainloop()