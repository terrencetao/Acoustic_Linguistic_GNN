python3 preprocess.py
python3 generate_similarity_matrix.py --sub_unit 15 --num_a 5 --ta 0 --tw 0.5
python3 build_kws_graph.py
python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/kws_graph.dgl
python build_kws_word_graph.py
python heterogenous_graph.py
python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 1000
