# Use an official Python image
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    x11-apps \
    portaudio19-dev \
    python3-gi \
    gir1.2-gtk-3.0 \
    gobject-introspection \
    libgirepository1.0-dev \
    build-essential \
    cmake \
    pkg-config \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-pulseaudio \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Upgrade pip before installing dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    Flask \
    pyaudio \
    numpy \
    google-cloud-speech \
    google-cloud-texttospeech \
    playsound==1.2.2 \
    google-generativeai \
    pygame \
    PyGObject \
    soundfile

# Set the environment variable for X11 forwarding
ENV DISPLAY=:0

# Run the application
CMD ["python", "main_play.py"]
