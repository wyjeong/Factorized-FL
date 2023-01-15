import pdb
from misc.utils import *
from data.loader import DataLoader
from modules.logger import Logger

class ServerModule:
    def __init__(self, args, sd, gpus):
        self.args = args
        self.gpu_id = gpus[0]
        self.sd = sd

    def aggregate(self, local_weights, ratio=None):
        st = time.time()
        aggr_theta = OrderedDict([(k,None) for k in local_weights[0].keys()])
        if isinstance(ratio, list):
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name]*ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1/len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        print(f'[server] aggregation done ({round(time.time()-st, 3)} s)')
        return aggr_theta

class ClientModule:
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args 
        self._args = vars(self.args)
        self.loader = DataLoader(self.args)
        self.logger = Logger(self.args, self.worker_id, self.gpu_id)
       
    def switch_state(self, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate(self):
        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(self.loader.te_loader):
                x_batch, y_batch = batch
                x_batch = x_batch.cuda(self.gpu_id)
                y_batch = y_batch.cuda(self.gpu_id)
                # y_hat, lss = self.validation_step(x_batch, y_batch, i)
                y_hat, lss = self.validation_step(x_batch, y_batch)
                pred.append(y_hat)
                target.append(y_batch)
                loss.append(lss)
            acc_1, acc_5 = self.accuracy(torch.stack(pred).view(-1, self.args.n_clss), torch.stack(target).view(-1))
        return acc_1.item(), acc_5.item(), np.mean(loss)

    @torch.no_grad()
    def validate(self):
        st = time.time()
        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(self.loader.va_loader):
                x_batch, y_batch = batch
                x_batch = x_batch.cuda(self.gpu_id)
                y_batch = y_batch.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(x_batch, y_batch)
                pred.append(y_hat)
                target.append(y_batch)
                loss.append(lss)
            acc_1, acc_5 = self.accuracy(torch.stack(pred).view(-1, self.args.n_clss), torch.stack(target).view(-1))
        return acc_1.item(), acc_5.item(), np.mean(loss)

    @torch.no_grad()
    def accuracy(self, preds, targets, k=(1,5)):
        with torch.no_grad():
            preds = preds.topk(max(k), 1, True, True)[1].t()
            correct = preds.eq(targets.view(1, -1).expand_as(preds))
            res = []
            for k_i in k:
                correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / targets.size(0)))
            return res

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log
        })
