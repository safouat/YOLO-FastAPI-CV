FROM python:3.10.12-slim

WORKDIR /fastapi

COPY requirement.txt .
RUN pip install  -r requirement.txt

COPY . .

USER root

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install dill
RUN useradd -m appuser
USER appuser



CMD ["uvicorn", "compvis:app", "--host", "0.0.0.0"]
