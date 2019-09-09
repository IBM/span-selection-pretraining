#!/bin/bash

#stop at first error, unset variables are errors
set -o nounset
set -o errexit

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="$SCRIPTDIR/.."

doc_dir=$2
output=$1

#java -cp ${SCRIPTDIR}/irsimple.jar com.ibm.research.ai.irsimple.AsyncWriter \
#  ${output} /data/wikipedia/wikipassagesindex 2>&1 > instgen.log &

# a larger corpus would require an even larger splitting, 20 is fine for Wikipedia
gen_split=20
for (( i=0; i<${gen_split}; i++ ))
do
python ${SCRIPTDIR}/sspt_gen_async.py \
  --doc_dir ${doc_dir} \
  --output ${output} \
  --world_rank ${i} --world_size ${gen_split}
done

touch ${output}/finished.txt
