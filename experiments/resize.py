from PIL import Image

img = Image.open('power_metric_result.png')
print('Original size:', img.size)

resized = img.resize((1400, 550), Image.LANCZOS)
resized.save('power_metric_result.png')
print('Done. New size:', resized.size)
