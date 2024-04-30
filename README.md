Traceback (most recent call last):
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 792, in <module>
    main(args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 747, in main
    trainVitExtractor(args.save_location, vit_model, train_data, val_data, args)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/Train_CNN_LSTM.py", line 402, in trainVitExtractor
    outputs = model(inputs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/jovyan/Desktop/Workflow/Create/Task_Recognition/ViT_LSTM.py", line 43, in forward
    x = self.transforms(x)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torchvision/transforms/transforms.py", line 137, in __call__
    return F.to_tensor(pic)
  File "/home/jovyan/conda-envs/createPytorchEnv/lib/python3.9/site-packages/torchvision/transforms/functional.py", line 141, in to_tensor
    raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")
TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>

