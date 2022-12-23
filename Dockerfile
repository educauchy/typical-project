FROM python:3.8-slim
MAINTAINER Vadim Glukhov <educauchy@gmail.com>

COPY requirements.txt /typical_project/requirements.txt

RUN /bin/bash -c "pip install -r /typical_project/requirements.txt"

COPY data/ /typical_project/data/
COPY src/ /typical_project/src/
COPY models/ /typical_project/models/
COPY reports /typical_project/reports

WORKDIR /typical_project

CMD ["python", "/typical_project/src/models/sklearn/train.py"]
