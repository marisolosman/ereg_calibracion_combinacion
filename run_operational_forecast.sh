k=$(date +"%Y")
l=$(date +%m)
declare -a combination=("wsereg" "wpdf")
declare -a weighting=("same" "pdf_int" "mean_cor")
declare -a variables=("prec" "tref")

for i in {1..7} ; do
	for var in "${variables[@]}" ; do
		python calibration.py ${var} ${l} ${i}
		for m in "${weighting[@]}" ; do
			python get_mme_parameters.py ${var} --IC ${l} --leadtime ${i} --weight_tech $m
				for j in "${combination[@]}" ; do
					python real_time_combination.py ${var} --IC ${k}"-"${l}"-01" --leadtime ${i} $j --weight_tech $m
				done
		done
			python plot_rt_forecast.py ${var} --IC ${k}"-"${l}"-01" --leadtime ${i} 
	done
done



