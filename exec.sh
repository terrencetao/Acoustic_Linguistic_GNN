
python3 preprocess.py --drop_freq 0 --drop_int 0

python3 generate_similarity_matrix_acoustic.py --sub_unit 500 --method fixed 



python3 weak_ML.py --epochs 10

for msa in fixed knn mixed ;do
 python3 build_kws_graph.py --num_n 10 --ta 0 --alpha 1 --method $msa
 
 for mh in fixed ML  mixed;do
   for msw in fixed mixed semantics;do
	python3 generate_similarity_matrix_word.py --tw 0.5  --method $msw # tw= threshold for word similarity

	 # ta=threshold for acoustic similarity ,  number of neighbord for each node in acoustic graph
	python build_kws_word_graph.py

	
	python heterogenous_graph.py --twa 0.1 --num_n 10 --method $mh # twa = treshold  for acoustic-linguistic probability, number link for each linguistic node

	python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/kws_graph.dgl
	python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 1000

	python eval_embedding.py --twa 0.1 --num_n_h 10 --mhg fixed  --num_n_a 10 --ta 0 --alpha 1 --tw 0.5  --msw $msw
   done
 done
done
