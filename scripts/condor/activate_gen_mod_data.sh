#!/bin/sh

. ../../../rvenv/bin/activate || exit 1
exec ./gen_mod_data.sh "$@"