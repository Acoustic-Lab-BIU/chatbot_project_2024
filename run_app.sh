#!/bin/bash

# Set your container name and other variables
CONTAINER_NAME="my-tk-app"
IMAGE_NAME="chatbot_project_2024-tk-app"
WORKDIR="/app"

# Define the Python script with proper escaping for special characters
PYTHON_SCRIPT_CONTENT=$(cat <<'EOF'
import os

# Set environment variable
os.environ['XAUTHORITY'] = '/root/.Xauthority'

# Add additional Python commands below
#print(f"XAUTHORITY is set:{ os.environ['XAUTHORITY']}")
os.system('timedatectl set-timezone Asia/Jerusalem')
# Run the main Python application
os.system('python3 /app/main_play_respeaker.py')
EOF
)

# Start the Docker container and execute the Python script
docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --env DISPLAY=$DISPLAY \
    --volume "$HOME/.Xauthority:/root/.Xauthority:rw" \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume "/home/pal/chatbot_project_2024:/app" \
    --volume "/etc/resolv.conf:/etc/resolv.conf" \
    --privileged \
    --net host \
    --workdir "$WORKDIR" "$IMAGE_NAME" \
    bash -c "echo \"$PYTHON_SCRIPT_CONTENT\" > /app/temp_script.py && python3 /app/temp_script.py"



