
from PIL import Image
im = Image.open("data/baker/baker.jpg")

a = im.resize((2048//2, 1360//2))

a.show()

