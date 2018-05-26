import lightnet as ln
import torch
from utils.utils import download_file

CONF_THRESH = 0.001
NMS_THRESH = 0.4
DEFAULT_SIZE = (416, 416)

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
DECAY = 0.0005
BATCH = 64
MINI_BATCH = 8

RESIZE = 3
RS_STEPS = []
RS_RATES = []


def _download_yolo_weights():
    remote_path = 'https://pjreddie.com/media/files/darknet19_448.conv.23'
    local_dir = './vendor/lightnet'
    return download_file(remote_path, local_dir)

def _build_yolo_model(classes: List[str], input_dim):
    weights_path = _download_yolo_weights()
    net = ln.models.Yolo(len(classes), weights_path, CONF_THRESH, NMS_THRESH)
    net.postprocess.append(ln.data.transform.TensorToBrambox(input_dim, classes))
    return net

class Trainer(ln.engine.Engine):
    batch_size = BATCH
    mini_batch_size = MINI_BATCH

    def __init__(self, model, dataloader, max_batches=10):
        self.max_batches = max_batches
        optim = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE/BATCH,
            momentum=MOMENTUM,
            dampening=0,
            weight_decay=DECAY*BATCH
        )
        super(VOCTrainingEngine, self).__init__(model, optim, dataloader, **kwargs)

    def start(self):
        self.add_rate('resize_rate', RS_STEPS, RS_RATES, RESIZE)
        self.dataloader.change_input_dim()

    def process_batch(self, data):
        data, target = data
        if self.cuda:
            data = data.cuda()
        data = torch.autograd.Variable(data, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.batch % self.resize_rate == 0:
            self.dataloader.change_input_dim()

    def quit(self):
        if self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_folder, f'final.pt'))
            return True

class YoloModel(ln.engine.Engine):
    """Wrapper around lightnet.models.Yolo."""
    def __init__(self, classes: List[str], input_dim=DEFAULT_SIZE):
        self.classes = classes
        self.input_dim = input_dim
        self.model = _build_yolo_model(classes, input_dim)
        self.cuda = False

    def score(self):
        pass

    def evaluate(self):
        output = self.model._forward()
        bramboxes = self.model.postprocess(output)

    def train(self, dataset):
        dataloader = ln.data.DataLoader(
            dataset,
            batch_size = MINI_BATCH,
            shuffle = True,
            drop_last = True,
            num_workers = WORKERS if self.cuda else 0,
            pin_memory = PIN_MEM if self.cuda else False,
            collate_fn = ln.data.list_collate,
        )
        trainer = Trainer(self.model, dataloader, max_batches=10)
        trainer()
