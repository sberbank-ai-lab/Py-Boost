# Py-boost: a research tool for exploring GBDTs

Modern gradient boosting toolkits are very complex and are written in low-level programming languages. As a result,

* It is hard to customize them to suit one’s needs 
* New ideas and methods are not easy to implement
* It is difficult to understand how they work

Py-boost is a Python-based gradient boosting library which aims at overcoming the aforementioned problems. 

**Authors**: [Anton Vakhrushev](https://kaggle.com/btbpanda), [Leonid Iosipoi](http://iosipoi.com/).


## Py-boost Key Features

**Simple**. Py-boost is a simplified gradient boosting library but it supports all main features and hyperparameters available in other implementations.

**Fast with GPU**. Despite the fact that Py-boost is written in Python, it works only on GPU and uses Python GPU libraries such as CuPy and Numba.

**Easy to customize**. Py-boost can be easily customized even if one is not familiar with GPU programming (just replace np with cp).  What can be customized? Almost everuthing via custom callbacks. Examples: Row/Col sampling strategy, Training control, Losses/metrics, Multioutput handling strategy, Anything via custom callbacks


## Installation

<span style="color:red"><strong>To be done:<strong> add something.</span>


## Quick tour

Py-boost is easy to use since it has similar to scikit-learn interface. 

Task (regression/classification) is determined by a loss function. It is the only argument required. Other optional arguments: MSELoss for simple/multitask regression, BCELoss for binary/multilabel classification, CrossEntropyLoss for multiclassification, Custom for your own task. Trained model may be saved via pickle.


<span style="color:red"><strong>To be done:<strong> add code.</span>

## Customization

Customization is made via Callback class. One can easily determine actions which need to be done before/after train process or each iteration. Methods get an info dict as input. It contains all the train and validation data and the model state. This data could be accessed and modified in any way.
  
<br/>  
<span style="color:red">
The following code illustrates how to build a custom column sampling strategy based on estimated feature importance. 
<br/> 
<strong>To be done:<strong> add code.
</span>


## Other Sber AI Lab Projects
LightAutoML: https://github.com/sberbank-ai-lab/LightAutoML  
AutoWoE: https://github.com/sberbank-ai-lab/AutoMLWhitebox  
RePlay: https://github.com/sberbank-ai-lab/RePlay  


