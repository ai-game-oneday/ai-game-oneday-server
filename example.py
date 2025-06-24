import requests

url = "https://api.retrodiffusion.ai/v1/inferences"
method = "POST"

headers = {
    "X-RD-Token": "rdpk-86e108fa1d830ec5a4c9e286cb0b8899",
}

payload = {
    "width": 64,
    "height": 64,
    "prompt": "A really cool corgi",
    "num_images": 1,
}

response = requests.request(method, url, headers=headers, json=payload)
print(response.text)
