FROM gw000/keras:2.1.4-py3-tf-cpu

# install dependencies from debian packages
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    python-matplotlib \
    python-pillow

RUN python3 -c "import tensorflow;print(tensorflow.__version__)"
RUN pip3 install --upgrade pip
RUN pip3 install Flask>=1.0.0
RUN pip3 install opencv-python
RUN pip3 install tqdm

WORKDIR /retail
COPY . /retail

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]
EXPOSE 8000




