FROM python:3.9.13-slim

RUN pip install --no-cache-dir scikit-learn==1.1.1 joblib==1.1.0

ARG task

COPY main.py .
COPY data/tfidf.pkl .
COPY data/classifiers/$task.pkl .

# to access $task in CMD
# https://stackoverflow.com/questions/35560894/is-docker-arg-allowed-within-cmd-instruction
ENV task $task

CMD python main.py infer $task tfidf.pkl $task.pkl
