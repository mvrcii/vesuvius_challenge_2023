from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


path = "deft-yogurt_20231210-234206_mode=MAX_layer=25-40_th=0.00_inverted.png"
processed = Image.open(path)

processed = processed.convert('RGBA')
datas = processed.getdata()

newData = []
print("start converting")
for item in tqdm(datas):
    alpha = 255 - item[0]
    newData.append((0, 0, 0, alpha))
processed.putdata(newData)

processed.save("transparent" + path)
print("done saving")
