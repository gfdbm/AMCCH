import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F




class MOCOAverage(nn.Module):
    # k = batchsize * batchsize
    # def __init__(self, inputSize, outputSize, image_q_encoder, text_q_encoder, image_k_encoder, text_k_encoder, K=16384,T=0.07, m=0.999, dim=128, use_softmax=True):
    def __init__(self, inputSize, outputSize, image_q_encoder, text_q_encoder, image_k_encoder, text_k_encoder, K=16384,
                 T=0.07, m=0.999, use_softmax=True):
        super(MOCOAverage, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.image_q_encoder = image_q_encoder
        self.text_q_encoder = text_q_encoder
        self.image_k_encoder = image_k_encoder
        self.text_k_encoder = text_k_encoder
        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, m]))
        self.register_buffer("queue", torch.randn(inputSize, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.image_q_encoder.parameters(), self.image_k_encoder.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
                self.text_q_encoder.parameters(), self.text_k_encoder.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def cal_similarity(self, F_I, F_T):
        batch_size = F_I.size(0)
        size = batch_size
        a1 = 0.6
        a2 = 0.6
        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = a1 * S_I + (1 - a1) * S_T

        S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        S2 = (S2 + S2.t()) / 2
        S = a2 * S1 + (1 - a2) * S2

        return S


    def forward(self, images, texts,GCN, y, idx=None, epoch=None):

        images_outputs = [self.image_q_encoder(im) for im in images]
        texts_outputs = [self.text_q_encoder(txt.float()) for txt in texts]
        l = torch.concat(images_outputs)
        ab = torch.concat(texts_outputs)
        batchSize = l.size(0)
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            images_momentum = [self.image_k_encoder(im) for im in images]
            texts_momentum = [self.text_k_encoder(txt.float()) for txt in texts]
            l_encoder = torch.concat(images_momentum)
            ab_encoder = torch.concat(texts_momentum)
            S = self.cal_similarity(l,ab)
            k = GCN(l,ab,S)
            k = (l_encoder + ab_encoder)/2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat(
                [torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])],
                dim=1).to(k.device).view([-1])
            k = k[inx]
            k = k.sign_()
            l_i_pos = torch.einsum("nc,nc->n", [l, k]).unsqueeze(-1)
            l_t_pos = torch.einsum("nc,nc->n", [ab, k]).unsqueeze(-1)
            l_i_neg = torch.einsum("nc,ck->nk", [l, self.queue.clone().detach()])
            l_t_neg = torch.einsum("nc,ck->nk", [ab, self.queue.clone().detach()])
        logits_img = torch.cat([l_i_pos, l_i_neg], dim=1)
        logits_txt = torch.cat([l_t_pos, l_t_neg], dim=1)
        logits_img /= self.T
        logits_txt /= self.T
        logits_img = logits_img.contiguous()
        logits_txt = logits_txt.contiguous()
        labels = torch.zeros(logits_img.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)
        return logits_img, logits_txt, labels, images_outputs, texts_outputs

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output






