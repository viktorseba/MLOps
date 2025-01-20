FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY dummy_stuff/requirements.txt requirements.txt
COPY dummy_stuff/main.py main.py
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "main.py"]
