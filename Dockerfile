FROM mmlab:latest


ENV TZ=Asia/Tehran



COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


WORKDIR /app
COPY . /app

CMD ["python3", "main.py"]
