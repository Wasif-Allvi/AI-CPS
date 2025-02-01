FROM busybox:latest

# Create necessary directories
RUN mkdir -p /tmp/learningBase/train
RUN mkdir -p /tmp/learningBase/validation

# Copy data files
COPY tmp/learningBase/train/training_data.csv /tmp/learningBase/train/
COPY tmp/learningBase/validation/test_data.csv /tmp/learningBase/validation/

# Copy README.md
COPY README.md /