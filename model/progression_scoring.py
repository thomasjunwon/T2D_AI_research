def progression_scoring(X,model_paths,scalers):
    import torch.nn as nn
    import torch
    import numpy as np
    from .MLP_model import LitModel1


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    

    all_probs = []
    
    for model_path, scaler in zip(model_paths, scalers):

        # 1. 모델 로드
        model = LitModel1.load_from_checkpoint(model_path)
        model.to(device)
        model.eval()

        # 2. test scaling
        X_test_scaled = scaler.transform(X)

        # 3. tensor 변환
        X_test_tensor = torch.tensor(
            X_test_scaled,
            dtype=torch.float32
        ).to(device)

        # 4. prediction
        with torch.no_grad():

            logits = model(X_test_tensor)

            # softmax probability
            probs = torch.softmax(logits/3, dim=1)

            probs = probs.cpu().numpy()

        all_probs.append(probs)

    # =========================
    # 평균 probability
    # =========================

    mean_probs = np.mean(all_probs, axis=0)

    # 최종 prediction
    y_pred = mean_probs[0][1]+2*mean_probs[0][2]

    return y_pred*50