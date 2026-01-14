import os

# Emotion dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

path = os.getcwd()

# Offline music library paths
emotion_folders = {
    0: f'{path}\\Music\\Angry',
    1: f'{path}\\Music\\Disgust',
    2: f'{path}\\Music\\Fear',
    3: f'{path}\\Music\\Happy',
    4: f'{path}\\Music\\Sad',
    5: f'{path}\\Music\\Surprise',
    6: f'{path}\\Music\\Neutral'
}

# Function to load songs from a folder
def load_songs_from_folder(folder_path):
    songs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            songs.append(os.path.join(folder_path, file_name))
    return songs

# Load all songs into a dictionary
all_songs = {}
for emotion, folder in emotion_folders.items():
    all_songs[emotion] = load_songs_from_folder(folder)

# Placeholder function to determine the emotion of a song
def determine_emotion(song):
    # In a real-world scenario, replace this with actual emotion detection logic
    return 3  # For demonstration purposes, return the integer for 'Happy'

# Recommend songs based on the emotion of a given song,
# ensuring that the last played song is not recommended again.
def recommend_songs_based_on_emotion(song, last_played_song=None):
    emotion = determine_emotion(song)
    if emotion in all_songs:
        recommended_songs = all_songs[emotion]
        # Filter out the last played song
        if last_played_song:
            recommended_songs = [s for s in recommended_songs if s != last_played_song]
        return recommended_songs
    else:
        return []

# Example usage
song_to_recommend = 'test'
last_played_song = None  # Initially, no song has been played

# Simulate playing and recommending songs
for _ in range(3):  # Let's simulate 3 rounds of song recommendations
    recommended_songs = recommend_songs_based_on_emotion(song_to_recommend, last_played_song)
    
    if recommended_songs:
        # Select the first song as the next song to play
        next_song_to_play = recommended_songs[0]
        print(f"Playing: {next_song_to_play}")
        last_played_song = next_song_to_play  # Update the last played song
    else:
        print("No more songs to recommend.")
