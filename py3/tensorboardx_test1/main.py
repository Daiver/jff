import math
import tensorboardX

writer = tensorboardX.SummaryWriter("./logs/")

for i in range(0, 20):
    writer.add_scalar("some_cool_scalar", i, global_step=i)
    writer.add_scalar("cool_scalar_too", math.sin(i), global_step=i)
