#!/bin/bash

# Create logs directory
mkdir -p logs

# Unit divisors specific to each dataset
declare -A UNIT_DIVISORS=( ["spoken_digit"]=10 ["google_command"]=8 ["yemba_command_small"]=13 )

#################################### CONFIGURATION #################################################
DATASETS=("spoken_digit" "google_command" "yemba_command_small")
UNITS=$(seq 500 500 3000)
METHOD_MMA="fixed"
METHOD_MSA="knn"
ALPHAS=(1.0)
NS=$(seq 0.5 0.1 1.0) #density
LAMB_VALUES=$(seq 1 0.1 2.0)
MHG_METHODS=("fixed" "dnn")
MSW_METHODS=("phon_count")
MGW_METHODS=("full")
TWAS=(0.1)
PS=(1 2 4) # proportion of number of link between the graphs
####################################################################################################

preprocess_dataset() {
  python3 preprocess.py --drop_freq 0.0 --drop_int 0.0 --feature mfcc
}

generate_acoustic_similarity() {
  dataset=$1
  unit=$2
  mma=$3

  outfile="saved_matrices/${dataset}_${mma}_${unit}.npy"
  if [ ! -f "$outfile" ]; then
    python3 generate_similarity_matrix_acoustic.py --sub_unit "$unit" --method "$mma" --dataset "$dataset"
  fi
  python3 weakDense.py --epochs 100 --method_sim $mma --sub_unit $unit --dataset $dataset
  python3 weak_ML2.py --epochs 100 --method_sim $mma --sub_unit $unit --dataset $dataset
}

generate_word_similarity() {
  dataset=$1
  unit=$2
  mma=$3
  msw=$4

  python3 generate_similarity_matrix_word.py --tw 0.1 --method "$msw" --dataset "$dataset" --sub_units "$unit" --method_sim "$mma"
}

build_homogeneous_graph() {
  dataset=$1
  unit=$2
  mma=$3
  msa=$4
  alpha=$5
  n=$6

  div=${UNIT_DIVISORS[$dataset]}
  num=$(echo "$n*($unit/$div - 1)" | bc | awk '{print int($0)}')

  for ko in $(seq 0 1 "$num"); do
    python3 build_kws_graph.py --num_n "$num" --k_out "$ko" --ta 0 --alpha "$alpha" --method "$msa" --dataset "$dataset" --sub_units "$unit" --method_sim "$mma"

    for lamb in $LAMB_VALUES; do
      graph_file="saved_graphs/$dataset/$mma/$msa/kws_graph_${num}_${ko}_${unit}.dgl"
      python3 gnn_model.py --input_folder '' --graph_file "$graph_file" --epochs 100 --lamb "$lamb" >> "logs/gnn_${dataset}_${unit}_${ko}.log" 2>&1

      build_heterogeneous_graph_and_eval "$dataset" "$unit" "$mma" "$msa" "$alpha" "$n" "$num" "$ko"
    done
  done
}

build_heterogeneous_graph_and_eval() {
  dataset=$1
  unit=$2
  mma=$3
  msa=$4
  alpha=$5
  n=$6
  num=$7
  ko=$8

  div=${UNIT_DIVISORS[$dataset]}

  for mhg in "${MHG_METHODS[@]}"; do
    for msw in "${MSW_METHODS[@]}"; do
      generate_word_similarity "$dataset" "$unit" "$mma" "$msw"
      for mgw in "${MGW_METHODS[@]}"; do
        python3 build_kws_word_graph.py --method "$mgw" --dataset "$dataset"

        for twa in "${TWAS[@]}"; do
          for p in $(seq "${PS[0]}" 2 "${PS[1]}"); do
         
            #num_n_h=$(echo "$n*($unit/$div - 1)/$p" | bc | awk '{print int($0)}')
            num_n_h=$(echo "($unit/$div - 1)/$p" | bc | awk '{print int($0)}')
            printf "NUM_N_H: %.2f\n" "$num_n_h"

            python3 heterogenous_graph.py --twa "$twa" --num_n "$num_n_h" --method "$mhg" --msw "$msw" \
              --sub_units "$unit" --method_sim "$mma" --method_acou "$msa" --dataset "$dataset" \
              --num_n_ac "$num" --k_out "$ko"

            graph_file="saved_graphs/$dataset/$mma/$msa/$mhg/$msw/hetero_graph_${num}_${ko}_${num_n_h}_${unit}.dgl"
            python3 gnn_heto_model.py --input_folder '' --graph_file "$graph_file" --epochs 100 --lamb $lamb \
              >> "logs/hetero_${dataset}_${unit}_${ko}.log" 2>&1
            python3 gnn_heto_link_pred_model.py --input_folder '' --graph_file "$graph_file" --epochs 100 --lamb $lamb \
              >> "logs/hetero_${dataset}_${unit}_${ko}.log" 2>&1

            python3 eval_embedding.py --mma "$mma" --twa "$twa" --num_n_h "$num_n_h" --mhg "$mhg" \
              --num_n_a "$num" --k_out "$ko" --ta 0.5 --alpha "$alpha" --tw 0.5 --msw "$msw" \
              --msa "$msa" --mgw "$mgw" --sub_unit "$unit" --drop_freq 0.0 --drop_int 0.0 \
              --dataset "$dataset" --lamb $lamb --density $n

            for add in dnn; do
              python3 induct_eval_embedding.py --mma "$mma" --twa "$twa" --num_n_h "$num_n_h" --mhg "$mhg" \
                --num_n_a "$num" --k_out "$ko" --ta 0 --alpha "$alpha" --tw 0.5 --msw "$msw" \
                --msa "$msa" --mgw "$mgw" --sub_unit "$unit" --drop_freq 0.0 --drop_int 0.0 \
                --dataset "$dataset" --add "$add" --k_inf "$num" --lamb $lamb --density $n
            done
          done
        done
      done
    done
  done
}

####################################################################################################

# MAIN EXECUTION
# Uncomment if needed
# preprocess_dataset

for dataset in "${DATASETS[@]}"; do
  for unit in $UNITS; do
    generate_acoustic_similarity "$dataset" "$unit" "$METHOD_MMA"

    for alpha in "${ALPHAS[@]}"; do
      for n in $NS; do
        build_homogeneous_graph "$dataset" "$unit" "$METHOD_MMA" "$METHOD_MSA" "$alpha" "$n"
      done
    done
  done
done

