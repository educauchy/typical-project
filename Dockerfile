FROM python:3.8-slim
MAINTAINER Vadim Glukhov <educauchy@gmail.com>

RUN apt-get update && \
	apt-get install

COPY requirements.txt /app/requirements.txt

RUN /bin/bash -c "pip install -r /app/requirements.txt"

COPY data/ /app/data/
COPY src/ /app/src/
COPY models/ /app/models/
COPY reports /app/reports

WORKDIR /app

CMD ["python", "/app/src/models/train.py"]
