# Pivotchain
Image-Segmentation

```
import requests
url = "http://localhost:9012/get-result"
response = requests.post( url,files={'file':open('12283150_12d37e6389_z.jpg','rb')})
print(response.text.encode('utf8'))
```
