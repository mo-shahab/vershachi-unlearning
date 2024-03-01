# Use Alpine Linux as the base image
FROM python:3.10.4-alpine3.14

# Install build tools and dependencies
RUN apk add --no-cache build-base python3-dev py3-pip

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for the framework
COPY ./vershachi /app/vershachi

# Copy the requirements file
COPY ./requirements-3-10.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-3-10.txt

# Install the framework in editable mode
RUN pip install --no-cache-dir -e /app/

# Specify the default command to run when the container starts
CMD [ "python", "run.py" ]
