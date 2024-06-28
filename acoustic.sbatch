#!/bin/bash
# Nom du job
#SBATCH --job-name=acoustic_linguistic_test
# Nombre de tâches
#SBATCH --ntasks=1
# Nombre de CPU par tâche
#SBATCH --cpus-per-task=10
# Durée maximale d'exécution du job (HH:MM)
#SBATCH -w cpu48
#SBATCH --time=100:01
# Nom du fichier de sortie (%x sera remplacé par le nom du job et %j par le numéro du job)
#SBATCH --output=output_%x.%j.out
# Types de notifications par email (BEGIN, END, FAIL, ou ALL pour toutes)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yannick.Yomie_Nzeuhang@univ-lemans.fr
#SBATCH --output=%x.%j.out

echo ./exec.sh
