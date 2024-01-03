from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


path = "20231106155351_inklabels_32_35.png"
processed = Image.open(path)

processed = processed.convert('RGBA')
datas = processed.getdata()

newData = []

print("start converting")
for item in tqdm(datas):
    # Check if the pixel is white
    if item[:3] == (255, 255, 255):
        # Change white to red, keep the original alpha
        newData.append((255, 0, 0, item[3]))
    else:
        # Make non-white pixels transparent
        newData.append((0, 0, 0, 0))
processed.putdata(newData)

processed.save("transparent" + path)
print("done saving")
