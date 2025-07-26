FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libfreetype6-dev \
    libfontconfig1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; \
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')"

COPY main.py .

RUN mkdir -p /app/input /app/output

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NLTK_DATA=/usr/local/share/nltk_data

CMD ["python", "main.py"]
