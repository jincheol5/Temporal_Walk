import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import CTDNE_Graph
from model import CTDNE_Link_Prediction
from model_train import ModelTrainer

"""
<< Test >> 
CTDNE
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. Link Prediction
            """
            ### 1. set model
            data=DataUtils.preprocess_graph(
                dataset_name=f"enron"
            )
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=CTDNE_Graph(graph_df=graph_df,bipartite=bipartite)
            model=CTDNE_Link_Prediction(
                embed_dim=kwargs["embed_dim"],
                latent_dim=kwargs["latent_dim"],
                window=kwargs["window"],
                graph=graph
            )

            ### 2. set data_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=200,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=200,shuffle=False)

            ### 3. train model
            model=ModelTrainer.train_link_prediction(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **kwargs
            )

            ### 4. evaulate model
            result=ModelTrainer.evaluate_link_prediction(
                model=model,
                data_loader=test_loader,
                **kwargs
            )
            print(f"Evaluate ACC: {result["acc"]}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"email-Enron")
    parser.add_argument("--embed_dim",type=int,default=4)
    parser.add_argument("--latent_dim",type=int,default=4)
    parser.add_argument("--window",type=int,default=3)

    # CTDNE
    parser.add_argument("--model_name",type=str,default=f"CTDNE")
    parser.add_argument("--walk_len",type=int,default=5)
    parser.add_argument("--min_walk_len",type=int,default=3)
    parser.add_argument("--n_context_window",type=int,default=100)
    parser.add_argument("--max_attempt",type=int,default=100)
    parser.add_argument("--edge_sampling",type=str,default=f"uniform")
    parser.add_argument("--neighbor_sampling",type=str,default=f"uniform")

    parser.add_argument("--walk_epoch",type=int,default=10)
    parser.add_argument("--epoch",type=int,default=1)
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--optimizer",type=str,default=f"adam")
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num,
        "dataset_name":args.dataset_name,
        "embed_dim":args.embed_dim,
        "latent_dim":args.latent_dim,
        "window":args.window,
        "model_name":args.model_name,
        "walk_len":args.walk_len,
        "min_walk_len":args.min_walk_len,
        "n_context_window":args.n_context_window,
        "max_attempt":args.max_attempt,
        "edge_sampling":args.edge_sampling,
        "neighbor_sampling":args.neighbor_sampling,
        "walk_epoch":args.walk_epoch,
        "epoch":args.epoch,
        "lr":args.lr,
        "seed":args.seed,
        "optimizer":args.optimizer
    }
    test_fn(**test_config)