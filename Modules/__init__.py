# import os
# import subprocess

# def convert_mp3_to_wav(input_file, output_file):
#     try:
#         # Construct the command to run ffmpeg
#         command = ['ffmpeg', '-i', input_file, output_file]
        
#         # Execute the command
#         subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
#         print(f"Conversion successful: {input_file} to {output_file}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error during conversion: {e}")
#     except FileNotFoundError:
#         print("ffmpeg not found. Please ensure ffmpeg is installed and accessible.")

# def convert_all_mp3_in_folder(folder_path):
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith('.mp3'):
#                 input_file = os.path.join(root, file)
#                 output_file = os.path.join(root, file.rsplit('.', 1)[0] + '.wav')
#                 convert_mp3_to_wav(input_file, output_file)

# # Music folders dictionary
# music_folders = {
#     0: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Angry',
#     1: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Disgusted',
#     2: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Fearful',
#     3: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Happy',
#     4: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Neutral',
#     5: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Surprised',
#     6: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Sad'
# }

# # Convert all MP3 files in each specified folder
# for folder_path in music_folders.values():
#     convert_all_mp3_in_folder(folder_path)




# import os

# def delete_mp3_files(folder_path):
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith('.mp3'):
#                 file_path = os.path.join(root, file)
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted: {file_path}")
#                 except Exception as e:
#                     print(f"Error deleting {file_path}: {e}")

# # Music folders dictionary
# music_folders = {
#     0: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Angry',
#     1: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Disgusted',
#     2: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Fearful',
#     3: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Happy',
#     4: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Neutral',
#     5: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Surprised',
#     6: 'D:\\Github\\Emotion-detection\\src\\Webpage\\Music\\Sad'
# }

# # Delete all MP3 files in each specified folder
# for folder_path in music_folders.values():
#     delete_mp3_files(folder_path)






# import os

# def get_file_paths(folder_path):
#     file_paths = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_paths.append(os.path.join(root, file))
#     return file_paths

# folder_path = r'D:\\Github\\Emotion-detection\\src\\Webpage\\Music'
# file_paths = get_file_paths(folder_path)

# for file_path in file_paths:
#     print(file_path)

# Python program demonstrating 
# Multiple selection in Listbox widget 

# from tkinter import *

# def select_emotion(selected_button, selected_color):
#     # Reset all buttons to their lighter shades
#     for button in buttons:
#         button.config(bg=lighter_colors[buttons[button]])
    
#     # Highlight the selected button with the bright color
#     selected_button.config(bg=selected_color)

# window = Tk()
# window.geometry('600x100')  # Adjust the size to fit the horizontal layout

# # Define a dictionary with emotions and their associated bright colors
# emotions = {
#     "Angry": "red",
#     "Disgust": "green",
#     "Fear": "purple",
#     "Happy": "yellow",
#     "Neutral": "gray",
#     "Sad": "blue",
#     "Surprise": "orange"
# }

# # Define lighter shades for the non-selected state
# lighter_colors = {
#     "Angry": "#ffcccc",
#     "Disgust": "#ccffcc",
#     "Fear": "#e5ccff",
#     "Happy": "#ffffcc",
#     "Neutral": "#d9d9d9",
#     "Sad": "#ccccff",
#     "Surprise": "#ffedcc"
# }

# # Create a frame to hold the buttons
# frame = Frame(window)
# frame.pack(expand=YES, fill="both")

# # Create a dictionary to hold the buttons
# buttons = {}

# # Create and pack a button for each emotion with its corresponding bright color initially
# for emotion, color in emotions.items():
#     button = Button(frame, text=emotion, bg=color, width=10,
#                     command=lambda b=Button, c=color: select_emotion(b, c))
#     button.pack(side=LEFT, padx=5)
#     buttons[button] = emotion  # Store the button and its emotion in the dictionary

# window.mainloop()
