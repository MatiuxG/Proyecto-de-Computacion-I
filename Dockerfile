#imagen oficial de python 3.10 slim (mas ligera que la full)
FROM python:3.10-slim

#evitamos que pip cache cosas en la imagen final y que python escriba .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

#dependencias del sistema necesarias para psycopg2 y lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libxml2-dev \
        libxslt1-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#primero copiamos solo requirements para aprovechar el cache de docker entre builds:
#mientras no cambien las dependencias, no hay que reinstalar
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

#ahora si, copiamos el resto del codigo
COPY . .

#el contenedor por defecto se queda dormido; el cron en LORCA (o el comando
#manual desde el dashboard admin de PC2) ejecutara los scripts cuando toque.
#para correr a mano: docker compose exec pc1 python run_all.py
CMD ["tail", "-f", "/dev/null"]
