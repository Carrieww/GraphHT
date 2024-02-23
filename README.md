# A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs
This repository contains the implementations of 1) the sampling-based hypothesis testing framework and 2) the hypothesis-aware samplers PHASE and its optimized version Opt-PHASE. 

[//]: # (The details of the framework and the samplers are described in the following paper:)

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

## Authors


* **Yun Wang** - Department of Computer Science, the University of Hong Kong

* **Chrysanthi Kosyfaki** - Department of Computer Science, the University of Hong Kong

* **Sihem Amer-Yahia** - CNRS, The Université Grenoble Alpes

* **Reynold Cheng** - Department of Computer Science, the University of Hong Kong

## Contact us
If you have any inquiry or bug report, please send emails to me at <a href="mailto:carrie07@connect.hku.hk">carrie07@connect.hku.hk</a>.


[//]: # (## License)

[//]: # ()
[//]: # (This project is licensed under the MIT License - see the [LICENSE.md]&#40;LICENSE.md&#41; file for details)

[//]: # (## Acknowledgments)

[//]: # ()
[//]: # (* Hat tip to anyone whose code was used)

[//]: # (* Inspiration)

[//]: # (* etc)
