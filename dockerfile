FROM rocm/pytorch:latest AS container
COPY . /yt_model
WORKDIR yt_model
RUN ["pip", "install", "-r", "requirements-dev.txt"]
# RUN python3 modelTraining.py
# COPY output .
# ENTRYPOINT ["python3", "modelTraining.py"]