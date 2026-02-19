FROM python:3.11-slim

WORKDIR /app

# dependências básicas (opcional, mas útil p/ builds)
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY data ./data
COPY scripts ./scripts

# outputs como volume no compose
RUN mkdir -p /app/outputs

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
