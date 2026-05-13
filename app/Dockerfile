FROM python:3.11-slim

WORKDIR /app

# PyTorch CPU-only — separate layer so Docker caches it independently
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    torch-geometric==2.5.3

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model directly — avoids malformed URL from `spacy download` in Docker
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

COPY app/ .

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
