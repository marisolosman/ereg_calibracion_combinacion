#!/bin/bash

LANG=en_UK
declare -a variables=("tref") #"prec")
declare -a wtech=("same" "pdf_int" "mean_cor")
declare -a ctech=("wpdf" "wsereg" "count")

for n in {1..12} ; do
	for k in {3..7} ; do
		for j in "${variables[@]}" ; do #loop sobre las variables a calibrar
			#python calibration.py ${j} ${n} ${k}
			for l in "${ctech[@]}" ; do #loop sobre manera de combinarr modelos
				if [ "${l}" = "count" ] ; then
					python combination.py ${j} ${n} ${k} ${l}
				else
					for m in "${wtech[@]}" ; do 
						if [ "${l}" = "wsereg" ] ; then 
							python combination.py ${j} ${n} ${k}  ${l} --weight_tech ${m}

						else 
							python combination.py ${j} ${n} ${k} ${l} --weight_tech ${m}

						fi
					done
				fi
			done
			python plot_forecast.py ${j} ${n} ${k}
		done
	done

done
