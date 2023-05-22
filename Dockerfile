FROM python:3.9.16-slim

# Update & install OpenCV.
RUN apt update && apt install -y \
  python3-opencv && \
  apt clean -y

# Install python deps.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Application entrypoint.
WORKDIR /app
ENTRYPOINT [ "python3", "src/server.py" ]