#!/bin/bash

#codigo para testear los diferentes parametros de la calibracion y la combinacion. Lo hago solo para
#no calibro ni combino CESM1 porque dejo de tener operativos sus modelos

LANG=en_UK
declare -a variables=("tref")
declare -a wtech=("same" "pdf_int" "mean_cor")
declare -a ctech=("wpdf" "wsereg" "count")

for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
	for k in {1..7} ; do # loop sobre el plazo de pronostico
		for n in {1..12} ; do #loop sobre condicion inicial
			python calibration.py ${j} ${n} ${k} --CV
			for l in "${ctech[@]}" ; do #loop sobre manera de combinarr modelos
				if [ "${l}" = "count" ] ; then
					python combination.py ${j} ${n} ${k} ${l}
					#python verification.py ${j} ${n} ${k} ${l}
				else
					for m in "${wtech[@]}" ; do 
						if [ "${l}" = "wsereg" ] ; then 
							python combination.py ${j} ${n} ${k}  ${l} --weight_tech ${m}
							#python verification.py ${j} ${n} ${k} 1 ${l} 1 --weight_tech ${m} 
						else 
							python combination.py ${j} ${n} ${k} ${l} --weight_tech ${m}
							#python verification.py ${j} ${n} ${k} 1 ${l} --weight_tech ${m} 
						fi
					done
				fi
			done
			#python plot_forecast.py ${j} ${n}
		done
	done

done
