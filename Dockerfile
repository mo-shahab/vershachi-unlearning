# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /vershachi-unlearning

# Copy the local directory contents into the container at /vershachi-unlearning
COPY . /vershachi-unlearning

# Install development requirements
RUN pip install -r requirements-dev.txt

# Set the maintainer label
LABEL Maintainer="moshahab"

# Run the sanity checks script
CMD ["python", "run.py"]
