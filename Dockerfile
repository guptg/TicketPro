FROM tensorflow/tensorflow

RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib

COPY /data/training_data.csv ./training_data.csv
COPY /data/validation_data.csv ./validation_data.csv

COPY training.py ./training.py