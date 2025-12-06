FROM rocm/pytorch:latest AS container
COPY . /yt_model
WORKDIR yt_model

VOLUME /mnt/output

RUN ["pip", "install", "-r", "requirements-dev.txt"]
RUN ["python", "-m", "spacy", "download", "en_core_web_sm"]

# ENTRYPOINT ["python", "run.py", "-o", "/mnt/output"]
ENTRYPOINT ["python", "modelTraining.py", "-o", "/mnt/output"]