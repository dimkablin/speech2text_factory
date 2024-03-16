FROM python:3.10

WORKDIR /app
COPY . /app
VOLUME src app/src

RUN apt-get update
RUN make build

EXPOSE 80
ENV NAME env_file
CMD ["python", "src/main.py"]
