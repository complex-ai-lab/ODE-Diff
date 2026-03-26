# ODE-Diff

`covid-19` folder includes code for semi-synthetic covid-19 experiments.

`dex` folder includes code for fully synthetic dexamethasone experiments.

## Running the Code

 The code requires conda3 (or miniconda3), and one CUDA capable GPU. The instructions below guide you regarding running the codes in this repository. 

### Environment & Libraries

The full libraries list is provided as a `requirements.txt` in this repo. Please create a virtual environment with `conda` or `venv` and run

~~~bash
(myenv) $ pip install -r requirements.txt
~~~

### Training 

For training, you can reproduce the experimental results of all benchmarks by runing

~~~bash
(myenv) $ python main.py --name diffpo --config_file ./Config/scenario-modeling.yaml --gpu 0 --train
~~~


### Generation
```bash
(myenv) $ python main.py --name diffpo --config_file ./Config/scenario-modeling.yaml --gpu 0 --sample 0 --milestone 10
```
