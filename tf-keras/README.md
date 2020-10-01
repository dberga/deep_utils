
## Dependencies and libraries
> Mainly you need python 3.6.8 and python3-pip
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo pip3 install --upgrade setuptools pip
```
> (Optional) Visualization of network
```
sudo apt-get install graphviz
```

> Install python libraries (mainly you need tensorflow) with:
```
sudo pip3 install -r requirements.txt
```
> If not working, force installation by adding the flag --ignore-installed  
> This has been tested with [cuda10](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork) and [cudnn7.5](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar). Check your tensorflow compatibility [here](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible) and your cuda/cudnn compatibility [here](https://developer.nvidia.com/rdp/cudnn-archive).


## Run 
```
python3 main.py
```

