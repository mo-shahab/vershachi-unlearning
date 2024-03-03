FROM ubuntu:20.04 AS builder
RUN apt-get update && apt-get install -y build-essential python3-dev python3-pip

FROM python:3.10-slim
COPY --from=builder /usr/bin/python3 /usr/bin/python3

# Install build tools and dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev python3-pip

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for the framework
COPY ./vershachi /app/vershachi

# Copy the pyproject.toml file
COPY ./pyproject.toml /app/

# Install Poetry and upgrade it
RUN pip install --no-cache-dir poetry && poetry self update

# Install dependencies using Poetry
RUN poetry install

# Run the installed dependencies
# RUN poetry build

# Specify the default command to run when the container starts
CMD [ "python", "run.py" ]
