######################################### DATA PROCESSING ###########################################################################################

#python3 preprocess.py --drop_freq 0.0 --drop_int 0.0 --feature mfcc  run this one time


for dataset in spoken_digit ;do
for unit in $(seq 200 500 5000);do
for mma in fixed ;do
python3 generate_similarity_matrix_acoustic.py --sub_unit $unit --method $mma --dataset $dataset


#python3 weakDense.py --epochs 100 --method_sim $mma --sub_unit $unit --dataset $dataset
#python3 weak_ML2.py --epochs 100 --method_sim $mma --sub_unit $unit --dataset $dataset

###################################################### Homogenous graph GNN ###########################################################################################################
for msa in knn ;do
for alpha in $(seq 1.0 1.0 1.0);do
 for n in $(seq 0.80 0.1 1.0);do #densite de 50% n in [0.1 ; 1]
   num=$(echo "$n*($unit/10 - 1)" | bc | awk '{print int($0)}')
 for ko in $(seq 1 1 $num);do # ko number of negatifs elements ko in [1 ... Snum]
 python3 build_kws_graph.py --num_n $num --k_out $ko  --ta 0 --alpha $alpha --method $msa --dataset $dataset  --sub_units $unit --method_sim $mma
 for lamb in $(seq 0.1 0.1 1); do
 python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/$dataset/$mma/$msa/kws_graph_"$num"_"$ko"_"$unit".dgl --epochs 100 --lamb $lamb
 
######################################################### Heterogeneous graph and Heterogenous GNN #####################################################################################
 for mhg in  dnn  fixed  ;do
   for msw in phon_count ;do #phon_suit phon_coo mixed semantics;do
	python3 generate_similarity_matrix_word.py --tw 0.1  --method $msw --dataset $dataset --sub_units $unit --method_sim $mma # tw= threshold for word similarity

	 # ta=threshold for acoustic similarity ,  number of neighbord for each node in acoustic graph
	for mgw in full ;do
	
	python build_kws_word_graph.py --method $mgw --dataset $dataset

	for twa in $(seq 0.1 0.1 0.1);do
	    for p in $(seq 2 2 4);do
	  num_n_h=$(echo "$n*($unit/10 - 1)/$p" | bc | awk '{print int($0)}')
	  printf "%.2f\n" "$num_n_h"
	
	python heterogenous_graph.py --twa $twa --num_n $num_n_h --method $mhg --msw $msw --sub_units $unit --method_sim $mma --method_acou $msa --dataset $dataset --num_n_ac $num --k_out $ko # twa = treshold  for acoustic-linguistic probability, number link for each linguistic node

	                                            # path to hetero_graph: saved_graphs/dataset/$mma/$msa/$mhg/$msw/
	python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/$dataset/$mma/$msa/$mhg/$msw/hetero_graph_"$num"_"$ko"_"$num_n_h"_"$unit".dgl --epochs 100 --lamb 1
	#python gnn_heto_sage.py --input_folder '' --graph_file saved_graphs/google_command/$mma/$msa/$mhg/hetero_graph_"$num"_"$unit".dgl --epochs 100
	#python gnn_heto_with_attention_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 100

############################################################# TEST #####################################################################################################
	python eval_embedding.py --mma $mma --twa $twa --num_n_h $num_n_h --mhg $mhg  --num_n_a $num --k_out $ko --ta 0.5 --alpha $alpha --tw 0.5  --msw $msw --msa $msa --mgw $mgw --sub_unit $unit --drop_freq 0.0 --drop_int 0.0 --dataset $dataset --lamb 1
	for add in dnn ;do
	python induct_eval_embedding.py --mma $mma --twa $twa --num_n_h $num_n_h --mhg $mhg  --num_n_a $num --k_out $ko --ta 0 --alpha $alpha --tw 0.5  --msw $msw --msa $msa --mgw $mgw --sub_unit $unit --drop_freq 0.0 --drop_int  0.0 --dataset $dataset --add $add --k_inf $num --lamb 1
	echo ""
        done
        done
        done
        
 done
done
done
done
done
done
done
done
done
done
done
