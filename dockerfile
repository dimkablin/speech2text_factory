FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN make build
EXPOSE 80
ENV NAME env_file
CMD ["python", "src/main.py"]
