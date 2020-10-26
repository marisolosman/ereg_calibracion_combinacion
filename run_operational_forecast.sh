#!/bin/bash

LANG=en_UK
declare -a variables=("prec" "tref")
declare -a wtech=("same" "pdf_int" "mean_cor")  # weighting
declare -a ctech=("wsereg" "wpdf")  # combination

readonly yellow=$(tput setaf 3)
readonly reset=$(tput sgr0)

usage() {
  echo -e " -y, --year   \t Year. The default value is the current year."
  echo -e " -m, --month  \t Month. The default value is the current month."
  echo -e " -o, --OW     \t Overwrite previous calibrations."
  echo -e " -h, --help   \t Display a help message and quit."
}

# process script inputs
while [[ $# -gt 0 ]]; do
  case $1 in
    -y|--year)  y=$2; shift 2;;
    -m|--month) m=$2; shift 2;;
    -o|--OW) OW="--OW"; shift 1;;
    -h|--help|*) usage; exit;;
  esac
done

# set default values when needed
readonly y=${y:-$(date +"%Y")}
readonly m=${m:-$(date +"%m")}

for var in "${variables[@]}" ; do
	for k in {1..7} ; do #loop over leadtime
		for j in "${ctech[@]}" ; do
			for w in "${wtech[@]}" ; do
        echo ${yellow}"$(date +'%D-%T') -- Running -- python real_time_combination.py ${var} --IC ${y}-${m}-01 --leadtime ${k} ${OW} ${j} --weight_tech ${w}"${reset}
				python real_time_combination.py ${var} --IC ${y}"-"${m}"-01" --leadtime ${k} ${OW} ${j} --weight_tech ${w}
			done
		done
		echo ${yellow}"$(date +'%D-%T') -- Running -- python plot_rt_forecast.py ${var} --IC ${y}-${m}-01 --leadtime ${k}"${reset}
		python plot_rt_forecast.py ${var} --IC ${y}"-"${m}"-01" --leadtime ${k} 
	done
done


