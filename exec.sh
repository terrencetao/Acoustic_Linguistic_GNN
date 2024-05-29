python3 preprocess.py --dim_init 10
python3 generate_similarity_matrix_acoustic.py --sub_unit 10 --num_n 2 --ta 0 
python3 generate_similarity_matrix_words.py --tw 0.5 
python3 build_kws_graph.py
python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/kws_graph.dgl
python3 weak_ML.py 
python build_kws_word_graph.py
python heterogenous_graph.py --twa 0.6 --num_n 5
python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 1000
