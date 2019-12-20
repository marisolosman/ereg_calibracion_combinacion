#!/bin/bash

#run verification of combined forecast

LANG=en_UK
declare -a variables=("prec")
declare -a wtech=("same" "pdf_int" "mean_cor")
declare -a ctech=("wpdf" "wsereg" "count")

for i in {1..7} # loop sobre el plazo de pronostico
do
	for n in {1..12} #loop sobre condicion inicial
	do
		for l in "${ctech[@]}" #loop sobre manera de combinarr modelos
		do
			if [ "${l}" = "count" ]
			then
				python verification.py prec ${n} ${i} ${l}
			else
				for m in "${wtech[@]}" #loop sobre forma de pesar modelos
				do
					python verification.py prec ${n} ${i} ${l} --weight_tech ${m} 
				done
			fi
		done
	done

done

