FROM tiangolo/uvicorn-gunicorn:python3.10


COPY ./requirements.txt /app/requirements.txt
COPY .  /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]