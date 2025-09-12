FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV TZ=Asia/Tehran
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY install.sh .
RUN chmod +x install.sh && ./install.sh

COPY . .

CMD ["python3", "main.py"]
