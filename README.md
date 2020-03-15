## Pivotchain Coding Challenge (Image-Segmentation)
### How to use !
- Clone github `https://github.com/Akshay-Udnur/Pivotchain.git`
- Change dir to Pivotchain `cd Pivotchain`
- Create virtual environment `virtualenv --python=python3.6 venv`
- Activate virtual environment `source venv/bin/activate`
- Install requirements using requirements.txt  `pip install -r requirements.txt`
- To start server run app.py file `python app.py`

-- First time app.py file downloads weight file. Due to that app.py takes 3-5 min to run at first time.

-- To get segmentation run following command. It will give rois, masks, class_ids, class_names, scores for all segmented objects. And also create segmented images in output folder.

```
import requests
url = "http://localhost:9012/get-result"
#iamge_path = '12283150_12d37e6389_z.jpg'
iamge_path = 'path to iamge'
response = requests.post( url,files={'file':open(iamge_path,'rb')})
print(response.text.encode('utf8'))
```
