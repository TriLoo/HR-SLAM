This folder includes some shell scripts.

* gpu_memory.sh

  This file can be used to monitor the usage of gpu memory.

  * Usage:

    ./gpu_memory.sh $1 $2 $3

	* $1: the number of iteration

	* $2: the name of log file

	* $3: the time step between two record

	The total monitor time is calculated by: $1 * $2
