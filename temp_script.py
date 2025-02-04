import os

# Set environment variable
os.environ['XAUTHORITY'] = '/root/.Xauthority'

# Add additional Python commands below
#print(fXAUTHORITY is set:{ os.environ[XAUTHORITY]})
os.system('timedatectl set-timezone Asia/Jerusalem')
# Run the main Python application
os.system('python3 /app/main_play_respeaker.py')
