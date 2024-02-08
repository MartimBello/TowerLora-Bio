from datasets import load_dataset, DownloadConfig


download_config = DownloadConfig(cache_dir='/mnt/data/martimbelo/TowerEval/data/training_data/')
dataset = load_dataset('Unbabel/TowerBlocks-v0.1', split='train', download_config=download_config)
print(dataset[0])
