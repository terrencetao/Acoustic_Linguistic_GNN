#!/bin/bash

# Usage: ./run_tasks.sh --unit <unit_value> --alpha <alpha_value> --num <num_value> --msa <msa_value> --msw <msw_value> --mgw <mgw_value> --mhg <mhg_value>

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sub_unit)
            unit="$2"
            shift 2
            ;;
        --alpha)
            alpha="$2"
            shift 2
            ;;
        --num)
            num="$2"
            shift 2
            ;;
        --msa)
            msa="$2"
            shift 2
            ;;
        --msw)
            msw="$2"
            shift 2
            ;;
        --mgw)
            mgw="$2"
            shift 2
            ;;
        --mhg)
            mhg="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if all required parameters are provided
if [[ -z "$unit" || -z "$alpha" || -z "$num" || -z "$msa" || -z "$msw" || -z "$mgw" || -z "$mhg" ]]; then
    echo "Missing required parameters."
    exit 1
fi

# Execute commands with provided parameters
python3 generate_similarity_matrix_acoustic.py --sub_unit "$unit" --method fixed

python3 weak_ML.py --epochs 10

python3 build_kws_graph.py --num_n "$num" --ta 0 --alpha "$alpha" --method "$msa"

python3 generate_similarity_matrix_word.py --tw 0.5 --method "$msw"

for mgw_type in empty full; do
    python build_kws_word_graph.py --method "$mgw_type"
done

python heterogenous_graph.py --twa 0.1 --num_n "$num" --method "$mhg" --msw "$msw"

python3 gnn_model.py --input_folder '' --graph_file saved_graphs/kws_graph.dgl --epochs 1000

python gnn_heto_model.py --input_folder '' --graph_file saved_graphs/hetero_graph.dgl --epochs 2000

