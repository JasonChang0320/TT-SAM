import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../model/model11_loss.csv")

train_loss = data.query("key=='train_loss'")
validation_loss = data.query("key=='val_loss'")

fig, ax = plt.subplots()
ax.plot(train_loss["step"], train_loss["value"], label="train")
ax.plot(validation_loss["step"], validation_loss["value"], label="validation")
ax.scatter(
    validation_loss["step"][validation_loss["value"].idxmin()],
    validation_loss["value"].min(),
    c="red",
    s=30,
)
ax.legend()
ax.set_ylabel("loss")
ax.set_xlabel("epoch")
# fig.savefig(f"model/model11_loss_curve.png",dpi=300)
