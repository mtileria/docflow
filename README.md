# docflow

This is the repository of the DocFlow analysis framework. 
DocFlow is an NLP framework to classify methods based on their documentation. 

## Getting started


```
1. Clone this repo: `git clone https://github.com/mtileria/docflow.git`
2. Create the docker image. Run the following command in the directory that contains the Dockerfile
    `docker build -t docflow_base .` 
3. change the path '/your/path/artifact' in the init_container.sh
4. run `./init_container.sh`
5. Go to http://localhost:8888/ or the link showed in the terminal
6. Run the commands in the extra-package.txt file. (This take some time to finish)
```


## Core Features

DocFlow includes:
- Android documentation crawler
- Sensitive methods classifier
- Semantic category classifier
- Semantic operations module 
- Script to reproduce results from the paper DocFlow: Extracting Taint Specifications from Software Documentation

## Usage


Run the notebook paper_results.ipynb in the folder __experiments__ and follow the instruction
Instruction on how to use DocFlow will be added soon

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

