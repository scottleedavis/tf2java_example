# tf2java_example

Experiment in models built by tensorflow in python, and loaded into java  

(Examples all use tensorflow 1.8)

## models built by python tensorflow

* [model variables](model/by_variables) -- [save_model_variables.py](python/save_model_variables.py) & [load_model_variables](python/load_model_variables.py)
* [model (graph)](model/by_graph) -- [save_model.py](python/save_model.py) & [load_model.py](load_model.py)


## models loaded by java

* [tensorflow (1.8)](java/tensorflow)  ( mvn compile exec:java -q )
* [dl4j (deeplearning4j)](java/dl4j) *todo*