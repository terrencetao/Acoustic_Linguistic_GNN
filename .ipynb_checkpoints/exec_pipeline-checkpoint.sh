#!/bin/bash

# Create logs directory
mkdir -p logs

# Unit divisors specific to each dataset
declare -A UNIT_DIVISORS=( ["timit"]=6102 ["spoken_digit"]=10 ["google_command"]=34 ["mini_speech_commands"]=8 ["yemba_command_small"]=13 )

#################################### CONFIGURATION #################################################
DATASETS=("timit_5_100" "timit" "mini_google_commands" "google_commands"  "spoken_digit" "yemba_command_small")
UNITS=$(seq 50000 100 60000)
METHOD_MMA="clique"
METHOD_MSA="filter"
ALPHAS=(1.0)
NS=$(seq 1 0.1 1.0) #density
LAMB_VALUES=$(seq 1 0.1 1)
MHG_METHODS=("fixed" "full_weighted")
MSW_METHODS=("phon_art" "phon_count" "allmini" "labse" "glove")
MGW_METHODS=("full")
TWAS=(0.1)
FEATURES=("wav2vec" "hubert" "wavlm" "yamnet" "vggish" "mel_spec" "mfcc")
PS=(0.5 1 2 4) # proportion of number of link between the graphs
####################################################################################################

preprocess_dataset() {
  python3 preprocess.py --drop_freq 0.0 --drop_int 0.0 --feature mfcc
}

generate_acoustic_similarity() {
  dataset=$1
  unit=$2
  mma=$3
  feature=$4

  outfile="saved_matrices/${dataset}_${mma}_${unit}.npy"
  if [ ! -f "$outfile" ]; then
    python3 generate_similarity_matrix_acoustic.py --sub_unit "$unit" --method "$mma" --dataset "$dataset" --feature "$feature"
  fi
  #python3 weakDense.py --epochs 50 --method_sim $mma --sub_unit $unit --dataset $dataset
  #python3 weak_ML2.py --epochs 50 --method_sim $mma --sub_unit $unit --dataset $dataset
}

generate_word_similarity() {
  dataset=$1
  unit=$2
  mma=$3
  msw=$4

  python3 generate_similarity_matrix_word.py --tw 0.7 --method "$msw" --dataset "$dataset" --sub_units "$unit" --method_sim "$mma"
}

build_homogeneous_graph() {
  dataset=$1
  unit=$2
  mma=$3
  msa=$4
  alpha=$5
  n=$6
  feature=$7

  div=${UNIT_DIVISORS[$dataset]}
  num=$(echo "$n*($unit/$div - 1)" | bc | awk '{print int($0)}')
  for msw in "${MSW_METHODS[@]}"; do
      generate_word_similarity "$dataset" "$unit" "$mma" "$msw" 
      for mgw in "${MGW_METHODS[@]}"; do
        generate_acoustic_similarity "$dataset" "$unit" "$mma" "$feature"
        
        python3 build_kws_word_graph.py --method "$mgw" --dataset "$dataset"

  #for ko in $(seq 0 1 "$num"); do
    ko=1
    python3 build_kws_graph.py --num_n "$num" --k_out "$ko" --ta 0 --alpha "$alpha" --method "$msa" --dataset "$dataset" --sub_units "$unit" --method_sim "$mma"

    for lamb in $LAMB_VALUES; do
      graph_file="saved_graphs/$dataset/$mma/$msa/kws_graph_${num}_${ko}_${unit}.dgl"
      #python3 gnn_model.py --input_folder '' --graph_file "$graph_file" --epochs 50 --lamb "$lamb" >> "logs/gnn_${dataset}_${unit}_${ko}.log" 2>&1

      build_heterogeneous_graph_and_eval "$dataset" "$unit" "$mma" "$msa" "$alpha" "$n" "$num" "$ko" "$feature"
    
  done
  
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
  feature=$9

  div=${UNIT_DIVISORS[$dataset]}

  for mhg in "${MHG_METHODS[@]}"; do
    
        for twa in "${TWAS[@]}"; do
          #for p in $(seq "${PS[0]}" 2 "${PS[1]}"); do
            p=1.0
            num_n_h=$(echo "$n*($unit/$div - 1)/$p" | bc | awk '{print int($0)}')
            num_n_h=$(echo "($unit/$div - 1)/$p" | bc | awk '{print int($0)}')
            #printf "NUM_N_H: %.2f\n" "$num_n_h"

            python3 heterogenous_graph.py --twa "$twa" --num_n "$num_n_h" --method "$mhg" --msw "$msw" \
              --sub_units "$unit" --method_sim "$mma" --method_acou "$msa" --dataset "$dataset" \
              --num_n_ac "$num" --k_out "$ko"

            graph_file="saved_graphs/$dataset/$mma/$msa/$mhg/$msw/hetero_graph_${num}_${ko}_${num_n_h}_${unit}.dgl"
        python3 gnn_heto_link_pred_model.py --input_folder '' --graph_file "$graph_file" --epochs 1000 --lamb $lamb --dataset $dataset\
        --mma "$mma" --twa "$twa" --mhg "$mhg" \
        --ta 0.5  --tw 0.5 --msw "$msw" \
         --msa "$msa" --mgw "$mgw" --sub_unit "$unit" \
         --dataset "$dataset" --lamb $lamb --feature $feature
              
              
            
            #python3 eval_embedding.py --mma "$mma" --twa "$twa" --num_n_h "$num_n_h" --mhg "$mhg" \
            #  --num_n_a "$num" --k_out "$ko" --ta 0.5 --alpha "$alpha" --tw 0.5 --msw "$msw" \
            #  --msa "$msa" --mgw "$mgw" --sub_unit "$unit" \
            #  --dataset "$dataset" --lamb $lamb --density $n --feature $feature

            
    done
  done
}

####################################################################################################

# MAIN EXECUTION
# Uncomment if needed
# preprocess_dataset

for dataset in "${DATASETS[@]}"; do
     for unit in $UNITS; do
    for alpha in "${ALPHAS[@]}"; do
      for feature in "${FEATURES[@]}"; do
        n=1.0
        build_homogeneous_graph "$dataset" "$unit" "$METHOD_MMA" "$METHOD_MSA" "$alpha" "$n" "$feature"
      done
    done
  done
done

