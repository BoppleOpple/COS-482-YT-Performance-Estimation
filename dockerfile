FROM rocm/pytorch:latest AS container

WORKDIR yt_model

COPY src ./src
COPY res ./res
COPY .vector_cache ./.vector_cache
COPY .env .
COPY requirements.txt .
COPY requirements-dev.txt .

VOLUME /mnt/output
VOLUME /mnt/thumbnails

RUN ["pip", "install", "-r", "requirements-dev.txt"]
RUN ["python", "-m", "spacy", "download", "en_core_web_sm"]

ENTRYPOINT ["python", "src/run.py", "-i", "/mnt/thumbnails", "-o", "/mnt/output"]
# ENTRYPOINT ["python", "src/modelTraining.py", "-i", "/mnt/thumbnails", "-o", "/mnt/output"]