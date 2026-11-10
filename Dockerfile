FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# Bake the model weights into the image so cold starts don't download them.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('piddnad/ddcolor_paper_tiny')"

COPY --chown=user app.py index.html ./
COPY --chown=user services/ddcolor/ddcolor services/ddcolor/ddcolor
COPY --chown=user services/ddcolor/basicsr services/ddcolor/basicsr

EXPOSE 7860
CMD ["python", "app.py"]
