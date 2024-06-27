#!/bin/bash
# Nom du job
#SBATCH --job-name=test
# Nombre de tâches
#SBATCH --ntasks=1
# Nombre de CPU par tâche
#SBATCH --cpus-per-task=10
# Durée maximale d'exécution du job (HH:MM)
#SBATCH --time=1:00
# Nom du fichier de sortie (%x sera remplacé par le nom du job et %j par le numéro du job)
#SBATCH --output=output_%x.%j.out
# Types de notifications par email (BEGIN, END, FAIL, ou ALL pour toutes)
#SBATCH --mail-type=ALL
# Adresse email pour recevoir les notifications
#SBATCH --mail-user=yynzeuhang@gmail.com
# La commande à exécuter
echo "Hello, world!"
