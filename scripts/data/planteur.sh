#!/bin/bash
# PLANTEUR (allenai/planteur-benchmark, ODC-BY): the files the loader reads (S2, S1, labels, metadata;
# ~120 GB of the 258 GB release: no Landsat, mosaics or embeddings) -> data/PLANTEUR. Plain curl, 12
# in parallel (the Hub allows ~3000 resolves / 5 min, so this takes >= 25 min). Resumable: finished
# files are skipped; rerun after a failure. Prints FAILED <path> per failed file.
#   REV=<commit> bash scripts/data/planteur.sh      (default: the revision the paper used)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DEST=data/PLANTEUR; B=https://huggingface.co/datasets/allenai/planteur-benchmark/resolve/${REV:-51b74a91424e}
{
  for f in metadata.geojson README.md NORM_S2_patch.json NORM_S1_patch.json splits/443.json \
           splits/loio.json splits/fewshot_pixels.json documentation/class_map.json \
           documentation/label_names.json documentation/colormap.txt \
           ANNOTATIONS_GEOTIFF/_crs_summary.json; do echo "$f"; done
  for i in $(seq 0 2749); do
    echo "ANNOTATIONS/TARGET_$i.npy"; echo "ANNOTATIONS_GEOTIFF/TARGET_$i.tif"
    echo "ANNOTATIONS_BALANCED/TARGET_$i.npy"; echo "DATA_S1/S1_$i.npy"; echo "DATA_S2/S2_$i.npy"
  done
} | xargs -P 12 -I{} sh -c '
  [ -s "$0/{}" ] && exit 0
  mkdir -p "$(dirname "$0/{}")"
  curl -sfL --retry 20 --retry-all-errors --retry-delay 5 -o "$0/{}.part" "$1/{}" \
    && mv "$0/{}.part" "$0/{}" || echo "FAILED {}"' "$DEST" "$B"
echo DONE
