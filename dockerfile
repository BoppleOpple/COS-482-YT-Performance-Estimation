FROM rocm/pytorch:latest AS container
COPY . /yt_model
WORKDIR yt_model

VOLUME /mnt/output

RUN ["pip", "install", "-r", "requirements-dev.txt"]
ENTRYPOINT ["python3", "run.py", "-o", "/mnt/output"]