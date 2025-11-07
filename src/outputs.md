RuntimeError                              Traceback (most recent call last)
Cell In8], [line 1
----> 1 main()

Cell In7], [line 10
      7     tokenizer.pad_token = tokenizer.eos_token
      9 print(f"Loading OpenWebText dataset split: {DATASET_SPLIT}...")
---> 10 dataset = load_dataset("openwebtext", split=DATASET_SPLIT)
     11 print(f"Dataset loaded with {len(dataset)} samples.")
     13 print("Tokenizing dataset...")

File /usr/local/lib/python3.10/dist-packages/datasets/load.py:1397, in load_dataset(path, name, data_dir, data_files, split, cache_dir, features, download_config, download_mode, verification_mode, keep_in_memory, save_infos, revision, token, streaming, num_proc, storage_options, **config_kwargs)
   1392 verification_mode = VerificationMode(
   1393     (verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS
   1394 )
   1396 # Create a dataset builder
-> 1397 builder_instance = load_dataset_builder(
   1398     path=path,
   1399     name=name,
   1400     data_dir=data_dir,
   1401     data_files=data_files,
   1402     cache_dir=cache_dir,
   1403     features=features,
   1404     download_config=download_config,
   1405     download_mode=download_mode,
   1406     revision=revision,
   1407     token=token,
   1408     storage_options=storage_options,
   1409     **config_kwargs,
   1410 )
   1412 # Return iterable dataset in case of streaming
   1413 if streaming:

File /usr/local/lib/python3.10/dist-packages/datasets/load.py:1137, in load_dataset_builder(path, name, data_dir, data_files, cache_dir, features, download_config, download_mode, revision, token, storage_options, **config_kwargs)
   1135 if features is not None:
   1136     features = _fix_for_backward_compatible_features(features)
-> 1137 dataset_module = dataset_module_factory(
   1138     path,
   1139     revision=revision,
   1140     download_config=download_config,
   1141     download_mode=download_mode,
   1142     data_dir=data_dir,
   1143     data_files=data_files,
   1144     cache_dir=cache_dir,
   1145 )
   1146 # Get dataset builder class
   1147 builder_kwargs = dataset_module.builder_kwargs

File /usr/local/lib/python3.10/dist-packages/datasets/load.py:1036, in dataset_module_factory(path, revision, download_config, download_mode, data_dir, data_files, cache_dir, **download_kwargs)
   1031             if isinstance(e1, FileNotFoundError):
   1032                 raise FileNotFoundError(
   1033                     f"Couldn't find any data file at {relative_to_absolute_path(path)}. "
   1034                     f"Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}"
   1035                 ) from None
-> 1036             raise e1 from None
   1037 else:
   1038     raise FileNotFoundError(f"Couldn't find any data file at {relative_to_absolute_path(path)}.")

File /usr/local/lib/python3.10/dist-packages/datasets/load.py:994, in dataset_module_factory(path, revision, download_config, download_mode, data_dir, data_files, cache_dir, **download_kwargs)
    986 try:
    987     api.hf_hub_download(
    988         repo_id=path,
    989         filename=filename,
   (...)
    992         proxies=download_config.proxies,
    993     )
--> 994     raise RuntimeError(f"Dataset scripts are no longer supported, but found {filename}")
    995 except EntryNotFoundError:
    996     # Use the infos from the parquet export except in some cases:
    997     if data_dir or data_files or (revision and revision != "main"):

RuntimeError: Dataset scripts are no longer supported, but found openwebtext.py

**** Solution

The runtime error you encountered with load_dataset("openwebtext", split=DATASET_SPLIT) indicating "Dataset scripts are no longer supported" suggests a change in how the Hugging Face datasets library handles certain datasets.

Explanation:
Hugging Face has deprecated dataset loading via scripts for some datasets, including openwebtext.

This means the dataset script file openwebtext.py is no longer usable with the new versions of the datasets library.

Instead, you are expected to use dataset versions provided in raw data formats or preprocessed Parquet formats or to download datasets from official repositories directly.

How to address this:
Check and update your datasets library to the latest version:

bash
pip install --upgrade datasets
Use a valid dataset loading method:

For OpenWebText, use official Hugging Face Datasets version or datasets that don’t rely on deprecated scripts.

Alternatively, use an alternative dataset with similar content or prepare your own dataset.

Check dataset docs or Hugging Face hub for updated usage:

Reference: https://huggingface.co/datasets/openwebtext

If you still want to use OpenWebText, consider loading from Parquet files or downloading dataset files directly and loading with load_dataset from local paths instead of script-based loading.

Summary:
The error is due to Hugging Face deprecating dataset script support for openwebtext. You should update your workflow to use newer dataset formats or sources supported by the latest library versions.