#!/bin/bash

#run verification of combined forecast

LANG=en_UK
declare -a variables=("prec")
declare -a wtech=("same" "pdf_int" "mean_cor")

for k in {1..7} # loop sobre el plazo de pronostico
do
	for n in {1..12} #loop sobre condicion inicial
	do
		python plot_model_weights.py prec ${n} ${k} 1  
	done
done


