#!/bin/sh

minite=0
remainder=0

iter=0

while [ "$iter" != $1 ]     # S1, the input number of iteration
do
    if [ "$remainder" -eq 0 ];then
        $(nvidia-smi | sed -n '1p' >> $2)     # $2 the name of log file
        $(nvidia-smi | grep 'MiB' | sed -n '1p' | awk '{print $9}' >> $2)
    fi
    iter=$(($iter+1))
    echo $iter

    minite=$(date +%M)
    remainder=$(($minite%$3))     # $3 the time step 
done

#$(nvidia-smi | sed -n '1p' >> $1)
#while [ "$remainder" == $0 ]
#do
#    minite=$(date +M)
#    remainder=$( ($minite%5) )
#
#    $(nvidia-smi | sed -n 'p1' >> test.txt)
#done

minite=$(date +%M)



#echo $minite
#
#iter=0
#
#echo $1
#
#while [ "$iter"  != $1 ]
#do
#    iter=$(($iter+1))   # cannot insert blankspace between $iter + and 1
#    echo $iter
#done

