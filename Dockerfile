FROM python:3.6
MAINTAINER Vadim Glukhov <educauchy@gmail.com>

RUN apt-get update && \
	apt-get install

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# executed when building image
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app
RUN ls -la /app/*

# executed when container starts
CMD ["python3", "/app/main.py"]