#!/bin/bash
LANG=en_UK
declare -a variables=("tref" ) #"prec")

for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
	for n in {1..12} ; do #loop sobre condicion inicial
		python plot_observed_category.py ${j} ${n} 
	done
done
