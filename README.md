# DQN-breakout
My first successful implementation of DQN on atari. (I had a much better version by switching the neural network implementation to PyTorch and using ddqn, but it was gone when I realize the files were not successfully copied into my USB drive after I erased the buggy linux mint system) On the brightside, my first successful version is stored on my windows systems

# A set of dependencies that works
python 3.8, gym==0.17.0 (important because the api of the older or newer versions are different), gym[atari], ale-py, tensorflow==2.0.0, matplotlib, opencv-python

# Training model
Execute `run.py` will start the training process. Don't use this to train because it was slow and its final average reward is low compared to my the PyTorch version I had earlier. I will recover the better version I had by replacing the network with pytorch when i have time

# Evaluate model
Run `eval.py` will run a pretrained model on the environment. It is fun to look at agents playing

# Results
The evaluation and training rewards are available in `train_rewards.npy` and `eval_rewards.npy`, you can plot it to see the learning curve (they are lists of rewards). Also, I used the code to extract highlight moments and are stored in `vlog`, and they are later converted to videos and stored in `videos` using the `view_log.py`


