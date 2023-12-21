from ultralytics import YOLO
import numpy as np

model = YOLO('./path/best.pt')  

results = model('./example.jpg')

names_dict = results[0].names
probs = (results[0].probs.data).tolist()

#print(results)
#print(probs)

print(names_dict[np.argmax(probs)])