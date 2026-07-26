import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric

class ModelTrainer:
    @staticmethod
    def train_link_prediction(
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            **kwargs
        ):
        """
        """

        """
        Train skip-gram
        """
        match kwargs["model_name"]:
            case "CTDNE":
                model.train_skipgram(
                    walk_len=kwargs["walk_len"],
                    min_walk_len=kwargs["min_walk_len"],
                    n_context_window=kwargs["n_context_window"],
                    max_attempt=kwargs["max_attempt"],
                    edge_sampling=kwargs["edge_sampling"],
                    neighbor_sampling=kwargs["neighbor_sampling"],
                    epoch=kwargs["walk_epoch"],
                    seed=kwargs["seed"]
                )
            case "ATDGEB":
                k_list=[2,4,6,8,10]
                model.train_skipgram(
                    k_list=k_list,
                    L=kwargs["L"],
                    min_points=kwargs["min_points"],
                    max_walk_len=kwargs["max_walk_len"],
                    n_sampling=kwargs["n_sampling"],
                    epoch=kwargs["walk_epoch"],
                    seed=kwargs["seed"]
                )
        """
        Train decoder
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )
        
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            model.graph.set_random_seed(epoch)
            for src,dst,ts in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                src=src.to(device) # [B,]
                dst=dst.to(device) # [B,]
                ts=ts.to(device) # [B,]

                ### negative sampling
                neg_dst=model.graph.random_negative_sampling(
                    src=src,
                    event_t=ts
                )
                pos_edge={
                    "src":src,
                    "dst":dst
                }
                neg_edge={
                    "src":src,
                    "dst":neg_dst
                }

                ### Label
                edge_label=TrainUtils.get_edge_label(
                    pos_edge_size=src.size(0),
                    neg_edge_size=neg_dst.size(0),
                    device=device
                ) # [2B,1]

                ### Predict
                pred_edge_logit=model(
                    pos_edge=pos_edge,
                    neg_edge=neg_edge
                ) # [2B,1]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_edge_logit,edge_label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            """
            validate model
            """
            ModelTrainer.evaluate_link_prediction(model=model,data_loader=val_loader,**kwargs)
        return model

    @staticmethod
    def evaluate_link_prediction(
            model:nn.Module,
            data_loader:DataLoader,
            **kwargs
        ):
        """
        Evaluate model
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=model.to(device)
        model.eval()
        model.graph.set_random_seed(kwargs["seed"])

        acc_list=[]
        with torch.no_grad():
            for src,dst,ts in tqdm(
                    data_loader,
                    desc=f"Evaluating..."
                ):
                src=src.to(device) # [B,]
                dst=dst.to(device) # [B,]
                ts=ts.to(device) # [B,]

                ### negative sampling
                neg_dst=model.graph.random_negative_sampling(
                    src=src,
                    event_t=ts
                )
                pos_edge={
                    "src":src,
                    "dst":dst
                }
                neg_edge={
                    "src":src,
                    "dst":neg_dst
                }

                ### Label
                edge_label=TrainUtils.get_edge_label(
                    pos_edge_size=src.size(0),
                    neg_edge_size=neg_dst.size(0),
                    device=device
                ) # [2B,1]

                ### Predict
                pred_edge_logit=model(
                    pos_edge=pos_edge,
                    neg_edge=neg_edge
                ) # [2B,1]

                ### Compute ACC
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_edge_logit,
                    label=edge_label
                )
                acc_list.append(batch_acc)
        acc=sum(acc_list)/len(acc_list)
        print(f"ACC: {acc}")
        return {
            "acc":acc
        }