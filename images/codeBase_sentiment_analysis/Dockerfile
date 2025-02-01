FROM busybox:latest

# Create necessary directory
RUN mkdir -p /tmp/activationBase

# Copy activation data
COPY tmp/activationBase/activation_data.csv /tmp/activationBase/

# Copy README 
COPY README.md /