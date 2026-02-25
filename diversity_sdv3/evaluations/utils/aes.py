
import math
import numpy as np
import torch
import clip

class AE(torch.nn.Module):
    def __init__(self, path=None, device="cuda"):
        super().__init__()
        self.ae_mlp = AE_MLP(768)
        s = torch.load(path)
        self.ae_mlp.load_state_dict(s)
        self.device = device
        self.ae_mlp.to(self.device)
        self.ae_mlp.eval()
        self.clip = ClipWrapper(self.device)

    def compute_img_embeddings(self, images):
        pr_imgs = [self.clip.preprocess(img) for img in images]
        pr_imgs = torch.stack(pr_imgs).to(self.device)
        return self.clip(pr_imgs).float()

    def forward(self, samples):
        img_embs = self.compute_img_embeddings(samples)
        img_embs = img_embs / torch.norm(img_embs, dim=-1, keepdim=True)

        # AE Score
        pred = self.ae_mlp(img_embs).cpu()

        if pred.ndim == 0:
            pred = pred.item()
        else:
            pred = pred.detach().cpu().numpy()
    
        return pred

class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14'):
        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, 
                                                        device, 
                                                        jit=False)
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)

class AE_MLP(torch.nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            #nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            #nn.ReLU(),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(64, 16),
            #nn.ReLU(),

            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)