import torch.nn as nn
import torch
from lightning.pytorch import LightningModule
from sklearn.metrics import classification_report, confusion_matrix


class MLPModel(nn.Module):
    def __init__(self,input_dim,inter_dim,dropout):
        super().__init__()
        #dim1=int(dim1)
        self.input_layer=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64,32),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(32,16),
            nn.GELU(),
            nn.Dropout(dropout),            
        )
        self.output_layer=nn.Linear(16,inter_dim)

    def forward(self,x):
        x=self.input_layer(x)
        output=self.output_layer(x)
        return output


class LitModel1(LightningModule):
    def __init__(self,  lr=1e-3,do=0.4, wd=1e-4):
        super().__init__()
        #dim1=int(dim1)
        self.save_hyperparameters()
        self.model = MLPModel(25,3,dropout=do)
        self.lr = lr
        self.wd=wd
        
        self.train_acc = []
        self.train_loss=[]
        self.valid_acc = []
        self.valid_loss=[]
        

    def forward(self, x):
        return self.model(x)
    
    def custom_loss(self,preds,targets):
        loss=nn.CrossEntropyLoss()        #HuberLoss: MSELoss, MAELoss의 단점을 보완하고 절충한 손실함수이다.
        final_loss=loss(preds,targets)
        return final_loss

    def training_step(self, batch,batch_idx):
        x, y = batch
        logits = self(x)    #forward 메서드 호출
        loss = self.custom_loss(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.custom_loss(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.custom_loss(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        self.y_preds.append(logits.argmax(dim=1).detach().cpu())
        self.y_trues.append(y.detach().cpu())
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer= torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.wd)
        return optimizer
        
    def predict_step(self, batch, batch_idx):
        x, _ = batch  # 정답(y)는 필요 없으므로 무시
        logits = self.forward(x)           # 예측 (logits)
        preds = torch.argmax(logits, dim=1)  # 가장 높은 점수의 클래스 선택

        return preds  # shape: [B]
    
    def predict_proba(self, x):
        """x: tensor of shape [B, features]"""
        self.eval()
        with torch.no_grad():
            logits = self.model(x)           # (B, 3)
            probs = torch.softmax(logits, dim=1)
        return probs

        

    def on_test_start(self):       #테스트 시작 시 한번만 실행
        self.y_preds = []
        self.y_trues = []

    def on_test_end(self):               #테스트 종료 시 한번만 실행
        preds = torch.cat(self.y_preds)     
        trues = torch.cat(self.y_trues)
        print("\nTest Report:")
        print(classification_report(trues, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(trues, preds))
        