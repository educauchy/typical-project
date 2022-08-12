FROM continuumio/miniconda3

COPY requirements.txt /typical_project/requirements.txt

RUN /bin/bash -c "cd /typical_project; python -m venv venv; source venv/bin/activate"
RUN /bin/bash -c "pip install -r /typical_project/requirements.txt"

COPY data/ /typical_project/data/
COPY src/ /typical_project/src/
COPY models/ /typical_project/models/
COPY reports/ /typical_project/reports/


CMD ["python", "/typical_project/src/models/sklearn/train.py"]