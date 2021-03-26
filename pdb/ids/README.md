## PDB IDs

This directory contains the IDs of the proteins from the PDB. They can be downloaded using the `pdb_pipeline.py` script at `src/utils/`.

### Query

The following snippet specifies the query used to obtain the data.
The query was carried out on the [PDB website](https://www.rcsb.org/search) using the Advanced Search settings.

```
Experimental Method = "X-RAY DIFFRACTION" AND Data Collection Resolution = [0-3] AND Entry Polymer Types = "Protein (only)" AND Macromolecule Name CONTAINS WORD "Protein" AND Macromolecule Name NOT CONTAINS WORD "DNA" AND Macromolecule Name NOT CONTAINS WORD "RNA" AND Macromolecule Name NOT CONTAINS WORD "Hybrid"
```

The query was carried out on March 25th 2021 at approximately 14:45 CET.

### Downloading the proteins

To download the proteins, run the following commands

```
./batch_download.sh -f rcsb_pdb_ids_1-25000.txt -o ../gz -p
./batch_download.sh -f rcsb_pdb_ids_25001-46258.txt -o ../gz -p
```

The output directory path (`-o ../gz`) corresponds to the default directory, which will also be used in the code. The `-p` options specifies that the downloaded files should have the `.pdb.gz` extension.