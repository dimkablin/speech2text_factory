FROM python:3.10

WORKDIR /app
COPY . /app
VOLUME app/src

RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra build-essential

RUN make build

EXPOSE 8000
ENV NAME env_file
CMD ["python", "src/main.py"]
