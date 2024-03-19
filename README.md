# Installation
To install all dependencies

```python
make build
```


and run locally 
```
python3 src/main.py
```


Or you can run a docker container
```bash
docker run -e BACKEND_URL="http://your_adress:your_port" -p your_port:8000 dimkablin/speech2text_factory
```
where ```BACKEND_URL``` is enviroment variable for greeting site ```docs/index.html```

by default the ```BACKEND_URL``` environment variable is set to ```http://localhost:8000```


# Example of usage
![alt text](docs/src/image.png)