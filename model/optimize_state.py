import torch
import numpy as np
import pandas as pd
from model.MLP_model import LitModel1
from model.progression_scoring import progression_scoring

import torch.cuda
import torch
from collections import defaultdict


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def optimize_with_mlp(
    model, scaler, x0_np, columns, device="cuda",
    lr=0.01, steps=300, lambda_reg=0.01, epsilon=5,
    fixed_features=['sex','age','edu','income','job','glu','hba1c','sbp','bmi','hdl','tg','ldl','wc'],
    clamp_dict = {
    'wk_smk': (0.0, 420.0),
    'wk_alc': (0.0, 40.0),
    'wk_mvpa_play': (0.0, 300.0), #mvpa는 최대 300분
    'wk_walk': (0.0, 1260.0),
    'wk_sleep': (360.0, 540.0),  #constraint
    'stress': (1.0, 4.0),
    'wk_break': (0.0, 6.0),
    'wk_lunch': (0.0, 6.0),
    'wk_dinner': (0.0, 6.0),
    'wk_veg1': (0.0, 21.0),
    'wk_veg2': (0.0, 21.0),
    'wk_fruit': (0.0, 21.0),
    }
):
    direction_constraints = {
    "wk_walk": "increase",
    "wk_mvpa_play":"increase",
    "stress": "decrease",
    "wk_alc": "decrease",
    "wk_smk": "decrease",
    "wk_veg1": "increase",
    "wk_veg2": "increase",
    }

    model.eval()
    x0_np = np.array(x0_np.values if hasattr(x0_np, "values") else x0_np).reshape(1, -1)

    x0 = torch.tensor(scaler.transform(x0_np), dtype=torch.float32, device=device)
    x = x0.clone().detach().requires_grad_(True)

    mask = torch.ones_like(x)
    for i, c in enumerate(columns):
        if c in fixed_features:
            mask[0, i] = 0.
    
    for _ in range(steps):

        if x.grad is not None:
            x.grad.zero_()

        logits = model(x)
        probs = torch.softmax(logits / 3, dim=1)
        score = (probs[:, 1] + 2 * probs[:, 2]) * 50
        loss = epsilon*score.mean() + lambda_reg * torch.norm(x - x0, p=1)
        loss.backward()

        with torch.no_grad():

            x -= lr * (x.grad * mask)

            for i, c in enumerate(columns):

                if c in fixed_features:
                    x[0, i] = x0[0, i]
                
                if c in direction_constraints:

                    if direction_constraints[c] =='increase':
                        x[0, i] = torch.maximum(x[0, i], x0[0, i])

                    elif direction_constraints[c]=='decrease':
                        x[0, i] = torch.minimum(x[0, i], x0[0, i])

                if clamp_dict and c in clamp_dict:

                    lo, hi = clamp_dict[c]

                    lo = (lo - scaler.mean_[i]) / scaler.scale_[i]
                    hi = (hi - scaler.mean_[i]) / scaler.scale_[i]

                    x[0, i] = torch.clamp(x[0, i], lo, hi)
            

    return scaler.inverse_transform(x.detach().cpu().numpy())

def get_feature_deltas(
    x0,
    x_opt,
    columns,
    fixed_features=['sex','age','edu','income','job','glu','hba1c','sbp','bmi','hdl','tg','ldl','wc'],
    tol=1e-6
    ):

    x0 = np.array(x0).reshape(-1)
    x_opt = np.array(x_opt).reshape(-1)

    deltas = {}

    for i, col in enumerate(columns):
        if col in fixed_features:
            continue

        diff = x_opt[i] - x0[i]

        if abs(diff) > tol:
            deltas[col] = float(diff)

    return deltas


def compute_total_decision(X,model_paths,scalers,lr=0.005, steps=400, lambda_reg=0.5, epsilon=5):

    x0 = X.iloc[[0]].values
    columns = X.columns.tolist()
    
    candidate_x_opts = []

    for i in range(5):
        best_model = LitModel1.load_from_checkpoint(model_paths[i])
        best_model.to(device)

        best_scaler = scalers[i]

        x_opt = optimize_with_mlp(
            model=best_model,
            scaler=best_scaler,
            x0_np=x0,
            columns=columns,
            device=device,
            lr=lr,
            steps=steps,
            epsilon=epsilon,
            lambda_reg=lambda_reg
        )
        
        candidate_x_opts.append(x_opt)

    candidate_scores = []
    
    for x_opt in candidate_x_opts:
        x_opt_df = pd.DataFrame(x_opt, columns=columns)
        score = progression_scoring(
            x_opt_df,
            model_paths,
            scalers
        )
        candidate_scores.append(float(score))

    best_idx = int(np.argmin(candidate_scores))
    best_x_opt = candidate_x_opts[best_idx]
            

    deltas = get_feature_deltas(
            x0=x0,
            x_opt=best_x_opt,
            columns=columns
        )

    return deltas
