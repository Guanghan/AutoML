# AutoML for Image Understanding

[Enter the AutoML: Papers and Resources](https://docs.google.com/document/d/193c3Eh_7C4Icvhfr1MxFlU7XrtP7WH4CyQUCIXmeUwk/edit)

[CF Space: More Info](https://cf.jd.com/pages/viewpage.action?pageId=369653056)

Docker Image: [idockerhub.jd.com/ailab_iu_repo/automl:v0.1](idockerhub.jd.com/ailab_iu_repo/automl:v0.1)

## To devs: recommended dev practice

#### 1. Documentation: [doc's doc](https://zh-sphinx-doc.readthedocs.io/en/latest/markup/toctree.html)
- pip install sphinx

##### 1.1 Docstring: [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- pip install sphinxcontrib-napoleon


##### 1.2 CMD: Generate API doc files
```
sphinx-apidoc -o rst ../src/utils -f
```

##### 1.3 CMD: Generate html files from documentation
```
make html
```

#### 2. Unit test
- pip install pytest / [Supported by PyCharm](https://www.jetbrains.com/help/pycharm/pytest.html#create-pytest-test)

#### 3. Formatter: [Google Style](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules/)
- pip install yapf



#### 4. Static Linting: code quality check
- pip install pylint

##### 4.1 CMD: Generate linting score and advice

```
pylint example.py 
```


#### 5. Logging 
pip install glog

#### 6. UML graph

- brew install graphviz

- pip install pyreverse

##### 6.1 CMD: Generate UML graph
```
pyreverse -o png -p AutoML src/
```

## To users: Getting Started

#### 1. Requirements
pip install PyYAML

#### 2. Example: Train DARTS on CIFAR10 

##### 2.1 Get docker image
```
docker pull idockerhub.jd.com/ailab_iu_repo/automl:v0.1
```

##### 2.2 Run docker container 
```
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 --shm-size=8g -it  -v /data:/data automl:latest
```

##### 2.3 Train DARTS
```
python src/core/pipeline.py > log.txt
```
