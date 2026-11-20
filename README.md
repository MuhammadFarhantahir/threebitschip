# Threebits neuroscience reproducibility materials

Muhammad Farhan Tahir, Threebits. Analysis code, derived summaries, figures, and embedded reference source for three research manuscripts. The human data analyses use public archival datasets. The controller and working-memory co-processor experiments are simulations.

## Setup

The research runs used Python 3.13.1 on arm64 macOS. Install the direct analysis dependencies and run commands from the repository root. A system C compiler is needed for the C99 decoder. Platform, dependency builds, and floating-point backends may affect exact reruns.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_datasets.py --help
```

`download_datasets.py` transfers only named public inputs and stores them in `data/`. `--list` displays destinations and sources without downloading. Asset SHA-256 digests are checked before use. The TMS-EEG and RAM analyses additionally read the required EEG ranges from the OpenNeuro public bucket. Upstream dataset terms and attribution apply.

| Source | Identifier and access | Downloader target and use |
| --- | --- | --- |
| OpenNeuro TMS-EEG and cognition, Biabani and Rogasch, CC0 | [ds008037 v1.0.1](https://doi.org/10.18112/openneuro.ds008037.v1.0.1) | `tmseeg`; scalp coordinates, with analysis scripts fetching the required data |
| OpenNeuro CCEP, van Blooijs et al., CC0 | [ds004080 v1.2.4](https://doi.org/10.18112/openneuro.ds004080.v1.2.4) | `ccep`; atlas MAT and electrode sidecars |
| OpenNeuro RAM FR3 and categorized FR3, CC0 | [ds005557 v1.0.0](https://doi.org/10.18112/openneuro.ds005557.v1.0.0), [ds005558 v1.0.0](https://doi.org/10.18112/openneuro.ds005558.v1.0.0) | `ram`; BIDS sidecars plus EEG range reads during analysis |
| HCP group connectivity and Schaefer parcellation | [Larivière et al.](https://doi.org/10.1038/s41592-021-01268-z), [Schaefer et al.](https://doi.org/10.1093/cercor/bhx179) | `hcp`; ENIGMA Toolbox matrices and atlas coordinates from commit-pinned sources |
| MICA-MICs, Royer et al., CC0 | [Data and methods paper](https://doi.org/10.1038/s41597-022-01682-y), [OSF data archive](https://osf.io/j532r/) | `mica`; v1.1 ZIP and selective extraction of the required Schaefer-100 matrices |
| MC_Maze, Neural Latents Benchmark, CC BY 4.0 | [DANDI 000128 v0.220113.0400](https://doi.org/10.48324/dandi.000128/0.220113.0400) | `mc-maze`; the public training asset used for the precision sweep |
| FALCON H1, Ye et al., CC BY 4.0 | [DANDI 000954 draft](https://dandiarchive.org/dandiset/000954/draft), [FALCON benchmark repository](https://github.com/snel-repo/falcon-challenge) | `falcon-h1`; 40 public assets pinned by ID and checksum. DANDI lists no published H1 version or dataset DOI. |

## Paper 1 — Causal brain simulation

```sh
python download_datasets.py hcp ccep tmseeg ram mica
python personal_brains.py data/mica_mics
python validate_simulator_ccep.py
python causal_brain_simulator.py
python validate_simulator_tmseeg.py --expanded
python stimulation_study.py data/mica_mics
python stimulation_followups.py data/mica_mics
python safe_closedloop_controller.py
python human_memory_closedloop.py
python human_memory_policy.py
```

The `mica` option downloads the 2.74 GB OSF archive and extracts the Schaefer-100 functional time series and structural matrices for 50 MICA-MICs models.

## Paper 2 — Low-bit FALCON decoder and firmware

```sh
python download_datasets.py mc-maze falcon-h1
python precision_sweep.py data/mc_maze_train.nwb
python falcon_precision_drift.py data/falcon_h1
python falcon_anchored_recal.py data/falcon_h1
python chip/bench_int4_gru.py
```

The FALCON results use the public calibration and validation data. The [submission workflow](falcon_submit/README.md) covers local bundle generation, container build, and official EvalAI evaluation. The generated C weight header contains packed model parameters derived from the FALCON H1 data credited above.

Run the firmware's synthetic-input host benchmark with:

```sh
mkdir -p build
cc -O3 -Wall -Wextra -DHOST_TEST chip/firmware_rp2350/main.c chip/int4_gru.c -o build/host_bench
./build/host_bench
```

An RP2350 build needs the [Raspberry Pi Pico SDK](https://github.com/raspberrypi/pico-sdk) and a supported Arm cross-toolchain. Set `PICO_SDK_PATH` to your local SDK checkout.

```sh
cmake -S chip/firmware_rp2350 -B build/pico2 -DPICO_BOARD=pico2
cmake --build build/pico2
```

The 62,944-byte parameter and lookup-table footprint and 3,404-byte static SRAM count describe the compiled source arrays. Physical-device timing, power, and total linked firmware size require separate measurements.

## Paper 3 — Working-memory EEG and co-processor simulation

```sh
python download_datasets.py tmseeg
python human_intelligence_wm_eeg.py
python replicate_tmseeg_g.py
python brain_ai_coprocessor.py
```

The working-memory analysis caches public inputs under `data/tmseeg_ds008037/`. The replication script uses the cognitive battery and discovery ERP caches created by the preceding analysis.

## Figures and outputs

```sh
python make_paper_figures.py
```

The figure builder consumes checked-in summaries at the repository root. Those summaries omit participant identifiers and unnecessary row-level results. Analysis commands can regenerate more detailed working outputs, which should be kept local. Raw EEG, neuroimaging, and NWB files are available from the upstream archives. The FALCON pickle is generated locally.
