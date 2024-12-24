FROM tensorflow/serving:latest
 
COPY ./serving_model /models
ENV MODEL_NAME=obesity_model