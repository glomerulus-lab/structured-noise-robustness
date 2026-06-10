#!/bin/sh

. ../../../rvenv/bin/activate || exit 1
exec ./mismatch_mod.sh "$@"