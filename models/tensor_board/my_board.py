import datetime

from torch.utils.tensorboard import SummaryWriter

# 创建 SummaryWriter 实例，日志存储在指定目录
def create_summary_writer():
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return SummaryWriter("runs/" + str(name))

writer = create_summary_writer()