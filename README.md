utils/data/dataloader.py", line 1346, in _next_data
    return self._process_data(data)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1372, in _process_data
    data.reraise()
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/_utils.py", line 722, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/DatasetGenerator.py", line 143, in __getitem__
    sequence_tensor,label_tensor = self.getBalancedSample()
  File "/home/jovyan/Desktop/ChallengeGithub/Central_Line_Challenge/Task_Recognition/DatasetGenerator.py", line 107, in getBalancedSample
    sample = self.sample_mapping[label][sample_idx]
KeyError: 0.0

