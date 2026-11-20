FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
RUN pip install --no-cache-dir falcon-challenge==1.0.2

WORKDIR /app
COPY baseline_decoder.py gru_decoder.py precision_sweep.py falcon_precision_drift.py falcon_anchored_recal.py ./
COPY chip/int4_gru.c chip/bench_int4_gru.py chip/
COPY falcon_submit/decoder.py falcon_submit/h1_int4_bundle.pkl falcon_submit/
RUN gcc -O3 -shared -fPIC chip/int4_gru.c -o falcon_submit/libint4_gru.so

ENV PREDICTION_PATH=/submission/submission.csv
ENV PREDICTION_PATH_LOCAL=/tmp/submission.pkl
ENV GT_PATH=/tmp/ground_truth.pkl
ENV EVALUATION_LOC=remote
ENV EVAL_DATA_PATH=/dataset/evaluation_data
ENV SPLIT=h1
ENV PHASE=minival
ENV BATCH_SIZE=1

CMD ["/bin/sh", "-c", "exec python falcon_submit/decoder.py --mode eval --evaluation \"$EVALUATION_LOC\" --model-path falcon_submit/h1_int4_bundle.pkl --split \"$SPLIT\" --phase \"$PHASE\" --batch-size \"$BATCH_SIZE\""]
