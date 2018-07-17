#!/bin/bash

# Got this from stack overflow
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -n "${PYROI_DIR}" ]; then
    if [ -d ${PYROI_DIR} ]; then
        # remove the previously defined PYROI_DIR and it's leading ':'
        PYTHONPATH=`echo $PYTHONPATH | sed -e 's#:'"${PYROI_DIR}"'##g'`
        # remove the previously defined PYROI_DIR without a leading ':'
        # couldn't get a \? escape on the : to work for some reason.
        PYTHONPATH=`echo $PYTHONPATH | sed -e 's#'"${PYROI_DIR}"'##g'`
    fi
fi

PYROI_DIR=$DIR
# If the python path already has directories in it, append it to the back
if [ -n "$PYTHONPATH" ]; then
    PYTHONPATH=$PYTHONPATH:$PYROI_DIR
else
    PYTHONPATH=$PYROI_DIR
fi

export PYROI_DIR
export PYTHONPATH
