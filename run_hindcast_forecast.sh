#!/bin/bash

#codigo para testear los diferentes parametros de la calibracion y la combinacion. Lo hago solo para
#no calibro ni combino CESM1 porque dejo de tener operativos sus modelos

LANG=en_UK
declare -a variables=("tref" "prec")
declare -a wtech=("same" "pdf_int" "mean_cor")
declare -a ctech=("wpdf" "wsereg" "count")

for j in "${variables[@]}" ; do #loop sobre las variables a calibrar, por ahora me enfoco en prec
	for n in {1..12} ; do # loop over IC
		for k in {1..7} ; do #loop over leadtime
			python calibration.py ${j} ${n} ${k}
			for l in "${ctech[@]}" ; do #loop sobre manera de combinarr modelos
				if [ "${l}" = "count" ] ; then
					python combination.py ${j} ${n} ${k} ${l}
				else
					for m in "${wtech[@]}" ; do 
						if [ "${l}" = "wsereg" ] ; then 
							python combination.py ${j} ${n} ${k} ${l} --weight_tech ${m}

						else 
							python combination.py ${j} ${n} ${k} ${l} --weight_tech ${m}

						fi
					done
				fi
			done
			python plot_forecast.py ${j} ${n} ${k}
		done
		python plot_observed_category.py ${j} ${n}
	done
done
