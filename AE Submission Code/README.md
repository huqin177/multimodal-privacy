# LLM-based PIE

This is the code released for the USENIX Security Symposium'25 paper "Evaluating LLM-based Personal Information Extraction and Countermeasures". 

## Assets

As promised in our Open Science section, we will share the Synthetic dataset and nny necessary code or configs to reproduce our experiments on the Synthetic and Court datasets. For the other three real-world datasets presented in our paper, as discussed in our Ethics and Open Science section, considering that these datasets contain sensitive real-world persons' information, we would not upload them to any public repository. Instead, we will send the Celebrity and Famous datasets to verified researchers per requests, as the persons' profiles in these datasets are in public domain.

Dataset: 

* Synthetic dataset: ./data/person100

Scripts:

* download\_court\_dataset.py: the script to download the Court dataset from Hugging Face (TAB benchmark)

* main.py: the script for main experiments

* main\_court.py: the script for experiments on Court dataset

* run.py: the script to run main.py and main\_court.py with different settings

* evaluate.py: the script to calculate evaluation results

* run\_evalaute.py: the script to run main.py and evaluate.py with different settings

* ./OpenLLMInfoExtraction: the main utilities for our experiments

Configurations:

* ./configs/model\_configs/

* ./configs/task\_configs/


## How to use


### Environment

The hardware environment for all models except for Flan is: Quadro RTX 6000, and the CUDA Version: 12.1. For Flan model, it fails to run on the above GPU, so we use RTX TITAN GPU instead. 

Before experiments, use conda to create an environment and activate:

```
conda env create -f environment.yml
conda activate llmpie
```

### Run the code

Now, Adjust the settings in run.py. For example, users can choose different models, defenses, etc. Users can run:

```
python3 run.py
```

After getting the results, users can adjust the settings in run\_evaluate.py and run:

```
python3 run_evaluate.py
```

The result will be located at ./logs/evaluate/

You are welcome to check the detailed code. Most of the code/configurations are with comments and self-explanatory. 


## Notes

First, some experiments in our paper were conducted using LLM APIs, including GPT-4, GPT-3.5, PaLM2, and Gemini. Users are responsible for obtaining the API access on their own. We encourage users to get APIs for GPT-models using Microsoft Azure, and the rest using Google AI Studio. These are where our APIs are documented and models in experiments are deployed. 

Second, we can't guarantee the API-based models will give exactly the same results as our paper. This is because we cannot control the RNG seeds. We can only control the RNG seeds for those open-source models. So, we encourage users to try experiments on those models. 

Third, again, we only share the Synthetic dataset here by default. For the other three real-world datasets presented in our paper, we (and the anonymous reviewers from USENIX Security Symposium) think, due to the dual nature of these data, it is not good to share them to anyone. As discussed in our Ethics and Open Science section, considering that these datasets contain sensitive real-world persons' information, we would not upload them to any public repository. Instead, we will send the Celebrity and Famous datasets to verified researchers per requests, as the persons' profiles in these datasets are in public domain. 

If you have any questions, please feel free to contact the authors of this paper. Thank you!
