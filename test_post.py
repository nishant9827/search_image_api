import requests

files = {'image': open('test.jpeg', 'rb')}  # Replace with any image you have
response = requests.post("http://192.168.29.6:5000/search", files=files)
print(response.status_code)
print(response.text)
