import pyaudio
import numpy as np

# Audio parameters
RATE = 16000  # Sampling rate
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio

def print_chunk_shape():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=11
    )

    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            # Read a chunk of audio data
            data = stream.read(CHUNK)
            
            # Convert the byte data to a NumPy array
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Print the shape of the audio array
            print(f"Chunk shape: {audio_array}")
    except KeyboardInterrupt:
        print("Stopped recording.")
    finally:
        # Close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    print_chunk_shape()
