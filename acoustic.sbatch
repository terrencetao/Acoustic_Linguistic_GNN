#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH -J acoustic_linguistic_gnn_train #job name
#SBATCH --mem=128G
#SBATCH --c 12
#SBATCH --w cpu48
#SBATCH --cpus-per-task=20 #ncpu on the same node
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yannick.Yomie_Nzeuhang@univ-lemans.fr
#SBATCH --output=%x.%j.out

./exec.sh
