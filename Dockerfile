FROM continuumio/miniconda3

COPY requirements.txt ./

RUN python3.7 -m venv venv
RUN source venv/bin/activate
RUN pip install -r requirements.txt

CMD ["python", "./src/models/sklearn", "--entry-point", "predict"]