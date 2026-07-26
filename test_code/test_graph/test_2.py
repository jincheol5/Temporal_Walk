import argparse
from utils import DataUtils
from graph import ATDGEB_Graph

"""
<< Test >> 
graph.ATDGEB_graph.ATDGEB_Graph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. 
            ATDGEB_Graph.detect_k_core_community()
            ATDGEB_Graph.detect_k_truss_community()
            ATDGEB_Graph.detect_k_clique_community()
            """
            dataset_name=f"enron"
            data=DataUtils.preprocess_graph(dataset_name=dataset_name)
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=ATDGEB_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )

            k_list=[10]
            result_1=graph.detect_k_core_community(w_list=k_list)
            result_2=graph.detect_k_truss_community(y_list=k_list)
            result_3=graph.detect_k_clique_community(z_list=k_list)
            print(f"dataset_name: {dataset_name}")
            print(f"k_list: {k_list}")
            print(f"k_core result:")
            print(result_1)
            print(f"k_truss result:")
            print(result_2)
            print(f"k_clique result:")
            print(result_3)

        case 2:
            """
            Test. 
            ATDGEB_Graph.generate_init_local_struct_vec()
            """
            dataset_name=f"enron"
            data=DataUtils.preprocess_graph(dataset_name=dataset_name)
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=ATDGEB_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )

            k_list=[2,4,6]
            stru=graph.generate_init_local_struct_vec(k_list=k_list)
            print(f"dataset_name: {dataset_name}")
            print(f"k_list: {k_list}")
            print(f"init local structure vector:")
            print(stru)

        case 3:
            """
            Test.
            ATDGEB_Graph.aggregate_local_struct_vec()
            """
            dataset_name=f"enron"
            data=DataUtils.preprocess_graph(dataset_name=dataset_name)
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=ATDGEB_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )

            k_list=[2,4,6]
            init_stru=graph.generate_init_local_struct_vec(k_list=k_list)
            stru=graph.aggregate_local_struct_vec(init_stru=init_stru,L=3)
            print(f"dataset_name: {dataset_name}")
            print(f"k_list: {k_list}")
            print(f"aggregated local structure vector:")
            print(stru)

        case 4:
            """
            Test.
            ATDGEB_Graph.generate_walks()
            """
            dataset_name=f"enron"
            data=DataUtils.preprocess_graph(dataset_name=dataset_name)
            graph_df=data["graph_df"]
            bipartite=data["bipartite"]
            graph=ATDGEB_Graph(
                graph_df=graph_df,
                bipartite=bipartite
            )
            graph.set_random_seed(seed=1)

            k_list=[2,4,6]
            L=3
            walks=graph.generate_walks(k_list=k_list,L=L)
            print(walks)

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num
    }
    test_fn(**test_config)