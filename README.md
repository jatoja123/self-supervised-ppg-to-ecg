# self-supervised-ppg-to-ecg
This repo contains the implementation of End-to-end non-invasive ECG signal generation from PPG signal: a self-supervised learning approach


### End-to-end non-invasive ECG signal generation from PPG signal: a self-supervised learning approach

## [Paper](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1694995/full) - [Repository](https://github.com/m1237/self-supervised-ppg-to-ecg)

<hr style="height:1px" />

<img src="./images/game-scenes.jpg"/>

### Abstract

Electrocardiogram (ECG) signals are frequently utilized for detecting important cardiac events, such as variations in ECG intervals, as well as for monitoring essential physiological metrics, including heart rate (HR) and heart rate variability (HRV). However, the accurate measurement of ECG traditionally requires a clinical environment, thereby limiting its feasibility for continuous, everyday monitoring. In contrast, Photoplethysmography (PPG) offers a non-invasive, cost-effective optical method for capturing cardiac data in daily settings and is increasingly utilized in various clinical and commercial wearable devices. However, PPG measurements are significantly less detailed than those of ECG. In this study, we propose a novel approach to synthesize ECG signals from PPG signals, facilitating the generation of robust ECG waveforms using a simple, unobtrusive wearable setup. Our approach utilizes a Transformer-based Generative Adversarial Network model, designed to accurately capture ECG signal patterns and enhance generalization capabilities. Additionally, we incorporate self-supervised learning techniques to enable the model to learn diverse ECG patterns through specific tasks. Model performance is evaluated using various metrics, including heart rate calculation and root mean squared error (RMSE) on two different datasets. The comprehensive performance analysis demonstrates that our model exhibits superior efficacy in generating accurate ECG signals (with reducing 83.9% and 72.4% of the heart rate calculation error on MIMIC III and Who is Alyx? datasets, respectively), suggesting its potential application in the healthcare domain to enhance heart rate prediction and overall cardiac monitoring. As an empirical proof of concept, we also present an Atrial Fibrillation (AF) detection task, showcasing the practical utility of the generated ECG signals for cardiac diagnostic applications. To encourage replicability and reuse in future ECG generation studies, we have made both the dataset and the code publicly available.

### Prerequisites
- Pytorch
- Numpy
- Scikit-learn
- Scipy
- Pandas
- h5py
- Matplotlib

## Data-Processing

Dataset files must be downloaded separately. The default loader in
`code/alyx_data.py` reads Who is Alyx? from `data/who-is-alyx` or
`data/who-is-alyx-2.1`. A DVC checkout must contain the actual physiological
JSON files, not just their `.dvc` pointers.

The loader supports Alyx JSON entries of the form
`"sample_index": [value, "receipt timestamp"]`, as well as timestamp-keyed
signals. For indexed recordings it reconstructs the sample clock at 130 Hz
(ECG) or 64 Hz (PPG), anchored at the first receipt timestamp. Bluetooth
receipt timestamps occur in batches and are not individual sample times.
Missing indices are treated as gaps. Unmarked sample loss and differences
in device delivery latency cannot be corrected from these files alone.

## Train 

Run `train.py` for the training. For evaluation run `eval.py`. Cross-validation should be done within train set.
Run `self_supervised_transformation.py` to make self-supervised ECG transformations.

Check the default dataset without starting training:

```powershell
.\.venv\Scripts\python.exe code/train.py --check-data
```

### Other datasets

The models and training loop are independent of the file format. Add a module
such as `code/my_data.py` exposing this function:

```python
def load_segments(data_dir, *, sampling_rate, window_seconds, overlap_ratio):
    # Read your format, align PPG/ECG, resample to sampling_rate, filter,
    # normalize each signal window to [-1, 1], and segment the recordings.
    # Return float32 arrays (N, T) and subject IDs of shape (N,).
    # T is normally round(sampling_rate * window_seconds).
    # Give every window/session of the same participant the same ID.
    return ppg_segments, ecg_segments, subject_ids
```

Select it without changing `train.py`:

```powershell
.\.venv\Scripts\python.exe code/train.py --data-loader my_data:load_segments --data-dir data/my-dataset --sampling-rate 128
```

`data_dir` is a `Path`, or `None` when omitted; each loader chooses its own
default directory. IDs can be integers or strings. The shared split requires
at least three participants and keeps each participant in a single split;
`--held-out-subject` selects the test participant. SSL uses the training ECG
windows for every dataset. Shared filtering and windowing helpers live in
`code/signal_preprocessing.py`.



### Citation
Please cite our paper below when using or referring to our work.
```
@article{yalcin_end_to_end_2026,
	author = {Yalcin, Murat and Latoschik, Marc Erich},
	year = {2026},
	month = {02},
	title = {End-to-end non-invasive ECG signal generation from PPG signal: a self-supervised learning approach},
	volume = {17},
	journal = {Frontiers in Physiology},
	url   = {https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1694995},
	issn	= {1664-042X},
	doi = {10.3389/fphys.2026.1694995}
}
```

