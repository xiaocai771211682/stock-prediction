FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
ENV FLASK_ENV=production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "html1_0:app"]