FROM python:3.11-alpine3.20
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./data /code/data
COPY ./train.py /code/train.py
COPY ./app /code/app
RUN python train.py
CMD ["fastapi", "run", "app/main.py", "--port", "80", "--workers", "4"]