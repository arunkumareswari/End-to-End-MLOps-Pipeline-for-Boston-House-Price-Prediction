FROM python:3.9

#set the working directory
WORKDIR /app

# copy the requirements.txt file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

EXPOSE 7860

# run the app
CMD ["python", "backend/app.py"]

