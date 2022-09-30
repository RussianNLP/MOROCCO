FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN pip install -U transformers==4.18.0 tokenizers datasets accelerate jsonlines

WORKDIR /bench

COPY main.py /bench
COPY rsg_funcs.py /bench

ENTRYPOINT ["python", "/bench/main.py", "infer"]

