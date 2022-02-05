# Decision tree building tool
This is a simple tool that lets you build, save and utilize decision trees. As for today, it's only capable of working with data (in csv format) that consists of **numerical attributes** and a **boolean class**. 

## Design choices
- tree building is done by top-down method
- stopping criteria: all cases belong to one class
- for every node new testing pool is created 
- test from the pool is picked proportionally to result's conditional entropy
- every subtree is swapped for a leaf if that doesn't reduce tree accuracy (tree trimming)

## Project files
#### `data_preparation.py`
Script created for easy preparation of data - splitting main data set into three data sets: training, trimming, and testing (3:1:1) with a preserved ratio of class representants. Each execution of this script will result in different data sets. 

#### `decision_tree.py`
Class representation of decision tree with its required utilities. An object of this class is an independent, fully functioning decision tree with additional data like data set size which tree was created from, and an index of the column in which category is stored. 

#### `main.py`
User-friendly way to interact with DecisionTree class and data preparation script. Created mainly for features presentation and DecisionTree usage reference. The use of the *Click* library provides a self-explanatory menu with help messages.  

Usage:
```
[mateusz@thinkpad ML_decision_tree]$ python3 main.py 
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build-tree
  build-tree-without-trim
  categorize-data
  prepare-data
  print-tree-info
  test-tree-accuracy
  trim-tree
```

## Testing
For testing reasons data set that is used for email spam classification was choosen and it can be found [here](https://archive.ics.uci.edu/ml/datasets/spambase). With use of this data set, measured tree accuracy was ~88%.

## Licence
MIT 
