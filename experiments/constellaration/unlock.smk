# This experiment-local no-op workflow lets `snakemake --unlock` operate on the repository's
# working directory without parsing a real run config or touching run outputs.
rule all:
    input: []
