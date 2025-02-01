FROM busybox:latest

# Create necessary directory
RUN mkdir -p /tmp/knowledgeBase

# Copy model files
COPY tmp/knowledgeBase/currentAiSolution.h5 /tmp/knowledgeBase/
COPY tmp/knowledgeBase/currentOlsSolution.pkl /tmp/knowledgeBase/

# Copy README 
COPY README.md /