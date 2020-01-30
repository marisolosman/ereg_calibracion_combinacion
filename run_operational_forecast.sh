k=$(date +"%Y")
l=$(date +%m)
declare -a combination=("wsereg" "wpdf")
declare -a weighting=("same" "pdf_int" "mean_cor")
declare -a variables=("prec" "tref")
for var in "${variables[@]}" ; do
	for i in {1..7} ; do
		for j in "${combination[@]}" ; do
			for m in "${weighting[@]}" ; do
			
				python real_time_combination.py ${var} --IC ${k}"-"${l}"-01" --leadtime ${i} $j --weight_tech $m
			done
		done
		python plot_rt_forecast.py ${var} --IC ${k}"-"${l}"-01" --leadtime ${i} 
	done
done


