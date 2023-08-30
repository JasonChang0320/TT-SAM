import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("model/2016 model train_val_loss.csv")

train_loss = data.query("key=='train_loss'")
valid_loss = data.query("key=='val_loss'")
fig, ax = plt.subplots(figsize=(7, 7))


ax.plot(train_loss["step"], train_loss["value"],label="train")
ax.plot(valid_loss["step"], valid_loss["value"],label="validation")
# final saved model
min_point = valid_loss["value"].idxmin()
ax.scatter(valid_loss["step"][min_point], valid_loss["value"][min_point],c="red",s=50)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Training loss curve")
ax.legend()
# fig.savefig("model/2016 model loss curve.png",dpi=300)
