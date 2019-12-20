#!/bin/bash
LANG=en_UK
declare -a variables=("prec" "tref")
for j in "${variables[@]}" #loop sobre las variables a calibrar, por ahora me enfoco en prec
do
	for n in {1..12} # loop sobre condicion inicial
	do
		for k in {1..7} #loop sobre plazo de pronostico
		do
		       python calibration.py ${j} ${n} ${k}

		done
	done
done

