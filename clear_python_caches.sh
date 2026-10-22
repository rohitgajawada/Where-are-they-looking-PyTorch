#!/usr/bin/env bash
# Remove historical Python bytecode/cache paths relative to the current directory.
# Run from the repository root; this preserves the original cleanup commands.
# It does not delete datasets, model weights, or training checkpoints.

rm -rf __pycache__
rm -rf datasets/__pycache__
rm -rf models/__pycache__
rm -rf models/layers/__pycache__
rm *.pyc
rm sfd/*.pyc
rm datasets/*.pyc
rm models/*.pyc
rm newlayers/*.pyc
