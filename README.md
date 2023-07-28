# docflow

This is the repository of the DocFlow analysis framework. 
DocFlow is an NLP framework to classify methods based on their documentation. 

## Getting started


```
1. Clone this repo: `git clone https://github.com/mtileria/docflow.git`
2. Create the docker image. Run the following command in the directory that contains the Dockerfile
    `docker build -t docflow_base .` 
3. change the path (required) '/your/path/artifact' and the name (optional) 'notebook'  in the init_container.sh file
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


Run the notebook [paper_results](experiments/paper_results.ipynb) in the folder __experiments__ and follow the instruction
To stop/restar the container you can run the following commands. The container name is notebook by defaul unless it was changed in the init_container.sh script. 

```
docker stop <container_name>
docker restart <container_name>
```


To use DocFlow as a standalone tool run the script [docflow.py](core/docflow.py)
Use -h to see the available options.   
Detailed instruction on how to use DocFlow and expected input/ouput will be added soon. 
To run the classifiers and search operations see the scripts in /core. These options will be integrated soon. 


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

