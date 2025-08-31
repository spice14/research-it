# research-it
build leann indexes - 

python -m src.index.build_from_url \
  "https://arxiv.org/html/2307.09218v3" \
  --index-path ./indexes/arxiv-2307-09218.leann \
  --chunk-size 700 --chunk-overlap 100
