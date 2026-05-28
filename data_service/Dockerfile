FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY data_service /app/data_service

EXPOSE 8001
CMD ["uvicorn", "data_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
