# Usage
```bash
python prepare_hestxenium_data.py \
	--data_dir <path to HEST-1k/data> \
	--output_dir <output direcotry> \
	--sample_id <HESTK-1k ID of Xenium samples to be processed> \
	--patch_size 4096 \
	--mode multi \
	--max_workers 32 \
	--batch_size 2
```