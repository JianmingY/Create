Traceback (most recent call last):
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py", line 696, in <module>
    main(args)
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py", line 654, in main
    trainResnet(foldDir, resnetModel,train_data,val_data,args,labelName=labelName)
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py", line 250, in trainResnet
    for step, (x, y) in enumerate(train_loader, start=epoch * len(train_loader)):
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 631, in __next__
    data = self._next_data()
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1346, in _next_data
    return self._process_data(data)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1372, in _process_data
    data.reraise()
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/_utils.py", line 722, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/pandas/core/indexes/base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 2606, in pandas._libs.hashtable.Int64HashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 2630, in pandas._libs.hashtable.Int64HashTable.get_item
KeyError: 13815

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/DatasetGenerator.py", line 142, in __getitem__
    sequence_tensor,label_tensor = self.getBalancedSample()
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/DatasetGenerator.py", line 107, in getBalancedSample
    imgFilePath = os.path.join(self.datacsv["Folder"][sample],self.datacsv["FileName"][sample])
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/pandas/core/series.py", line 1121, in __getitem__
    return self._get_value(key)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/pandas/core/series.py", line 1237, in _get_value
    loc = self.index.get_loc(label)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 13815
