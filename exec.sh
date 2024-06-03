
#python3 preprocess.py --drop_freq 0.5 --drop_int 0.3

python3 generate_similarity_matrix_acoustic.py --sub_unit 300 

for msw in semantics phon_count mixed;do
	python3 generate_similarity_matrix_word.py --tw 0.5  --method $msw # tw= threshold for word similarity

	python3 build_kws_graph.py --num_n 10 --ta 0 --alpha 10 # ta=threshold for acoustic similarity ,  number of neighbord for each node in acoustic graph
	python build_kws_word_graph.py

	#python3 weak_ML.py --epochs 100
	python heterogenous_graph.py --twa 0.1 --num_n 15 --method fixed # twa = treshold  for acoustic-linguistic probability, number link for each linguistic node

	python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/kws_graph.dgl
	python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 1000

	python eval_embedding.py --twa 0.1 --num_n_h 15 --mhg fixed  --num_n_a 10 --ta 0 --alpha 10 --tw 0.5  --msw $msw
done
