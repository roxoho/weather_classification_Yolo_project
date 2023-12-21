import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the csv file
results = pd.read_csv('./runs/classify/train/results.csv')

plt.figure()

plt.plot(results['                  epoch'],results['             train/loss'], label='train loss')
plt.plot(results['                  epoch'],results['               val/loss'], label='val loss', c='r')
plt.grid()
plt.title('loss vs epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(results['                  epoch'],results['  metrics/accuracy_top1'], label='train acc')
plt.grid()
plt.title('accuracy vs epoch')
plt.show()