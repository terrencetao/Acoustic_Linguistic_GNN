#python3 preprocess.py --drop_freq 0.5 --drop_int 0.3 --feature spec  run this one time

for unit in $(seq 100 100 2000);do
python3 generate_similarity_matrix_acoustic.py --sub_unit $unit --method fixed 



python3 weak_ML.py --epochs 10

for msa in fixed mixed ;do
for alpha in $(seq 1.0 0.5 2.0);do
 for num in $(seq 20 10 50);do
 python3 build_kws_graph.py --num_n $num --ta 0 --alpha $alpha --method $msa
 
 for mhg in fixed ML  mixed;do
   for msw in phon_coo phon_suit phon_count mixed semantics;do
	python3 generate_similarity_matrix_word.py --tw 0.5  --method $msw # tw= threshold for word similarity

	 # ta=threshold for acoustic similarity ,  number of neighbord for each node in acoustic graph
	for mgw in empty full ;do
	python build_kws_word_graph.py --method $mgw

	
	python heterogenous_graph.py --twa 0.1 --num_n $num --method $mhg --msw $msw # twa = treshold  for acoustic-linguistic probability, number link for each linguistic node

	python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/kws_graph.dgl --epochs 1000
	python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 2000

	python eval_embedding.py --twa 0.1 --num_n_h $num --mhg $mhg  --num_n_a $num --ta 0 --alpha $alpha --tw 0.5  --msw $msw --msa $msa --mgw $mgw --sub_unit $unit --drop_freq 0.5 --drop_int 0.3
        done
 done
done
done
done
done
done
