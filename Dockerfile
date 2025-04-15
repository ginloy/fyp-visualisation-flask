FROM ghcr.io/astral-sh/uv:debian


# Set working directory
WORKDIR /app

# Copy your project files
COPY pyproject.toml ./
COPY uv.lock ./
RUN uv sync

COPY static ./static
COPY cleaned.pq ./
COPY *.py ./

# # Install dependencies (uv creates a virtual environment automatically)
# RUN uv venv && \
#   uv pip install gunicorn && \
#   uv pip install .

# Set environment variables (optional)
# ENV PYTHONUNBUFFERED=1
ENV PORT=8050

# Expose the port Dash will run on
EXPOSE ${PORT}

# Run the Dash app using Gunicorn
CMD ["uv", "run", "gunicorn", "-w", "8", "-b", "0.0.0.0:8050", "main:app"]
