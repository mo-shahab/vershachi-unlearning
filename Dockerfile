# Use Alpine Linux as the base image
FROM python:3.10.4-alpine3.14

# Install build tools and dependencies
RUN apk add --no-cache build-base python3-dev py3-pip

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
RUN poetry build

# Specify the default command to run when the container starts
CMD [ "python", "run.py" ]
