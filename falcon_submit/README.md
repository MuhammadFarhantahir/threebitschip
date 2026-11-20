# FALCON H1 integer decoder bundle

The C99 runtime uses packed 4-bit weights and integer arithmetic. The submission runner can select adapters trained using the public held-out calibration data in [FALCON H1](https://dandiarchive.org/dandiset/000954/draft), distributed by Ye, Collinger, and Gaunt under CC BY 4.0.

From the repository root, in the analysis Python environment:

```sh
python download_datasets.py falcon-h1
python falcon_submit/decoder.py --mode train \
  --data-dir data/falcon_h1 --out falcon_submit/h1_int4_bundle.pkl
docker build --platform linux/amd64 -t threebits-h1-int4 -f falcon_submit/Dockerfile .
```

The pickle is generated locally from public data. Load bundles you have generated yourself. To regenerate the microcontroller header from that bundle, run `python chip/export_c_header.py`.

Use the submission tab of the [official FALCON EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2319/overview) for the current team, Docker, and phase instructions. The [organizers' evaluator documentation](https://github.com/snel-repo/falcon-challenge) describes local minival and remote test evaluation. The results reported here come from the public data.
