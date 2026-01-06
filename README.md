# A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation and experimental data for the paper **A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs**, accepted to **PVLDB 2024**.

📄 **Paper**: [Link to paper](https://arxiv.org/pdf/2403.13286)

This repository contains the implementations of 
1) the sampling-based hypothesis testing framework 
2) the hypothesis-aware samplers PHASE and its optimized version Opt-PHASE.

## Install and Run

1. Download the repository

```
git clone https://github.com/Carrieww/GraphHT.git
```
All graph data shall be stored in `\datasets`. Here we include a `graph.pkl` for MovieLens dataset for illustration.  

2. Install required packages

```
pip install -r requirements.txt
```
3. Run the framework

The preprocessed dataset can be found here on OneDrive: <a href="https://connecthkuhk-my.sharepoint.com/:f:/g/personal/carrie07_connect_hku_hk/EsIJtppoMWxGgsYpXvRb7c0B1tidq4n5XH43MvKxgSHWfw?e=v48xjz">link</a>.

You can either run:
```
python main.py --sampling_method "PHASE"
```
or specify the sampling method in `run.sh` and run
```
bash run.sh
```
in the terminal.

If you want to specify the dataset, sampling budget, and hypothesis, you can specify them in `config.py` and run using the above two lines of code.

## Output files
All output files are stored in the folder `/result/one_sample_log_and_results_*`. The two output files are 
1. **\*.txt:** a table seperated by tab storing the accuracy, time, p-value, and confidence interval results at the specified sampling budgets. 
2. **\*.log:** logger information

If you want to get plots for accuracy, time, p-value, and confidence interval versus the sampling budget, you can edit the parameters in `makePlot.py` and run it. The plot will be saved to the same folder as the txt and log files.

[//]: # (## Contributing)

[//]: # ()
[//]: # (Please read [CONTRIBUTING.md]&#40;https://gist.github.com/PurpleBooth/b24679402957c63ec426&#41; for details on our code of conduct, and the process for submitting pull requests to us.)

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{wang2024sampling,
  title={A Sampling-Based Framework for Hypothesis Testing on Large Attributed Graphs},
  author={Wang, Yun and Kosyfaki, Chrysanthi and Amer-Yahia, Sihem and Cheng, Reynold},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={11},
  pages={3192--3200},
  year={2024},
  publisher={VLDB Endowment}
}
```

**For questions or issues, please refer to the paper or contact the authors.**