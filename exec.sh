#python3 preprocess.py --drop_freq 0.0 --drop_int 0.0 --feature mfcc  run this one time

for dataset in google_command; do
for unit in $(seq 3000 1000 10000);do
for mma in fixed ;do
python3 generate_similarity_matrix_acoustic.py --sub_unit $unit --method $mma --dataset $dataset



python3 weakDense.py --epochs 100 --method_sim $mma --sub_unit $unit --dataset $dataset

for msa in knn ;do
for alpha in $(seq 2.0 1.0  2.0);do
 #for n in $(seq 0.02 0.01 0.1);do #densite de 50%
   num=$(echo "0.05*($unit/10 - 1)" | bc | awk '{print int($0)}')
 python3 build_kws_graph.py --num_n $num --ta 0.3 --alpha $alpha --method $msa --dataset $dataset  --sub_units $unit --method_sim $mma
 
 for mhg in dnn ;do
   for msw in phon_count;do #phon_suit phon_coo mixed semantics;do
	python3 generate_similarity_matrix_word.py --tw 0.5  --method $msw --dataset $dataset --sub_units $unit --method_sim $mma # tw= threshold for word similarity

	 # ta=threshold for acoustic similarity ,  number of neighbord for each node in acoustic graph
	for mgw in full ;do
	
	python build_kws_word_graph.py --method $mgw --dataset $dataset

	for twa in $(seq 0.6 0.1 0.6);do
	num_n_h=$(echo "0.25*($unit/10 - 1)" | bc | awk '{print int($0)}')
	python heterogenous_graph.py --twa $twa --num_n $num_n_h --method $mhg --msw $msw --sub_units $unit --method_sim $mma --method_acou $msa --dataset $dataset --num_n_ac $num # twa = treshold  for acoustic-linguistic probability, number link for each linguistic node

	python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/$dataset/$mma/$msa/kws_graph_"$num"_"$unit".dgl --epochs 100
	python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/$dataset/$mma/$msa/$mhg/hetero_graph_"$num_n_h"_"$unit".dgl --epochs 100
	#python gnn_heto_sage.py --input_folder '' --graph_file saved_graphs/google_command/$mma/$msa/$mhg/hetero_graph_"$num"_"$unit".dgl --epochs 100
	#python gnn_heto_with_attention_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 100

	python eval_embedding.py --mma $mma --twa $twa --num_n_h $num_n_h --mhg $mhg  --num_n_a $num --ta 0.5 --alpha $alpha --tw 0.5  --msw $msw --msa $msa --mgw $mgw --sub_unit $unit --drop_freq 0.0 --drop_int 0.0 --dataset $dataset
	python induct_eval_embedding.py --mma $mma --twa $twa --num_n_h $num_n_h --mhg $mhg  --num_n_a $num --ta 0 --alpha $alpha --tw 0.5  --msw $msw --msa $msa --mgw $mgw --sub_unit $unit --drop_freq 0.0 --drop_int  0.0 --dataset $dataset
        done
 done
#done
done
done
done
done
done
done
done
