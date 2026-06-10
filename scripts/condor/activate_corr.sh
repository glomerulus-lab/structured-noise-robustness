#!/bin/sh

. ../../../rvenv/bin/activate || exit 1
exec ./it_works_corr.sh "$@"