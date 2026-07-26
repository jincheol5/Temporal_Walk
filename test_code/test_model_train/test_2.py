import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import ATDGEB_Graph
from model import ATDGEB_Link_Prediction
from model_train import ModelTrainer

"""
<< Test >> 
ATDGEB
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
            graph=ATDGEB_Graph(graph_df=graph_df,bipartite=bipartite)
            model=ATDGEB_Link_Prediction(
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
            ModelTrainer.evaluate_link_prediction(
                model=model,
                data_loader=test_loader,
                **kwargs
            )

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

    # ATDGEB
    parser.add_argument("--model_name",type=str,default=f"ATDGEB")
    parser.add_argument("--L",type=int,default=2)
    parser.add_argument("--min_points",type=int,default=2)
    parser.add_argument("--max_walk_len",type=int,default=10)
    parser.add_argument("--n_sampling",type=int,default=1)
    
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
        "L":args.L,
        "min_points":args.min_points,
        "max_walk_len":args.max_walk_len,
        "n_sampling":args.n_sampling,
        "walk_epoch":args.walk_epoch,
        "epoch":args.epoch,
        "lr":args.lr,
        "seed":args.seed,
        "optimizer":args.optimizer
    }
    test_fn(**test_config)