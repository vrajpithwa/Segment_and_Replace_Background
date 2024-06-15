import requests
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}
import io
from PIL import Image
def query(payload):
	print('paylod')
	print(payload)
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
# image_bytes = query({
# 	"inputs": "Beautiful background with greenery with a tree ",
# })
# image = Image.open(io.BytesIO(image_bytes))

# Convert to PNG or JPG (choose one format)
# image.save('visible_image.png')  # Save as PNG

# You can access the image with PIL.Image for example



# byteImgIO = io.BytesIO()
# byteImg = Image.open("/Users/vrajpithwa/Desktop/Development/HCD/Unvileo/img.png")
# byteImg.save(byteImgIO, "PNG")
# byteImgIO.seek(0)
# byteImg = byteImgIO.read()
# # io.BytesIO(image_bytes).getvalue()

# image = Image.open(io.BytesIO(image_bytes))
# print(image)