FROM tiangolo/uvicorn-gunicorn:python3.10


COPY ./requirements.txt /app/requirements.txt
COPY .  /app

RUN ls -la /app

RUN pip install --no-cache-dir --upgrade --default-timeout=900 -r requirements.txt

WORKDIR /app
