# Pivotchain Coding Challenge (Image-Segmentation)
### How to use !
clone github ```https://github.com/Akshay-Udnur/Pivotchain.git```
change dir to Pivotchain ```cd Pivotchain```
create virtual environment ```virtualenv --python=python3.6 venv```
activate virtual environment ```source venv/bin/activate```
install requirements using requirements.txt ```pip install -r requirements.txt```
to start server run app.py file ```python app.py```

To get segmentation run following command. It will give rois, masks, class_ids, class_names, scores for all segmented objects. And also create augmented image in output folder.

```
import requests
url = "http://localhost:9012/get-result"
#iamge_path = '12283150_12d37e6389_z.jpg'
iamge_path = 'path to iamge'
response = requests.post( url,files={'file':open(iamge_path,'rb')})
print(response.text.encode('utf8'))
```
