import os
import glob
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser(description='Compare the runs')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
args = parser.parse_args()

scalars_to_compare = ['i_roc', 'p_roc', 'p_pro', 'loss']

datasets = set()
exp_types = set()

for tf_event in glob.glob(args.logdir + '/**/events.out.tfevents.*', recursive=True):
	if "wide_resnet50_2-mvtec2d-wi-OOM-wi-MOM" in tf_event:
		continue
	ea = event_accumulator.EventAccumulator(tf_event)
	file_path = Path(tf_event)
	dataset = file_path.parent.name
	datasets.add(dataset)
	exp_type = file_path.parts[1]
	exp_types.add(exp_type)
	ea.Reload()
	loss_len = len(ea.Scalars('loss'))

datasets_map = {dataset: i for i, dataset in enumerate(datasets)}
exp_types_map = {exp_type: i for i, exp_type in enumerate(exp_types)}
scalars_to_compare_map = {scalar: i for i, scalar in enumerate(scalars_to_compare)}

values = np.zeros((len(exp_types), len(datasets), loss_len))

for tf_event in glob.glob(args.logdir + '/**/events.out.tfevents.*', recursive=True):
	if "wide_resnet50_2-mvtec2d-wi-OOM-wi-MOM" in tf_event:
		continue
	ea = event_accumulator.EventAccumulator(tf_event)
	file_path = Path(tf_event)
	dataset_idx = datasets_map[file_path.parent.name]
	exp_type_idx = exp_types_map[file_path.parts[1]]

	ea.Reload() # loads events from file
	tags = ea.Tags() # print available tags

	scalars_keys = tags["scalars"]
	tensors_keys = tags["tensors"]

	scalars = ea.Scalars('loss')

	comp_value = np.array([x.value for x in scalars])
	values[exp_type_idx][dataset_idx] = comp_value


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
exp_mean_trajs = values.mean(axis=1)
# for exp_type in exp_types:
exp_type = "freq_step"
i = exp_types_map[exp_type]
# traj = values[i][datasets_map["wood"]]
trajectory_loss = exp_mean_trajs[i]
size = len(trajectory_loss)

diff_list = [0]*2
diff2_list = [0]*2
for i in range(2, size, 1):
	traj = trajectory_loss[:i]
	x_size = len(traj)
	x = np.arange(x_size)

	axis_ratio = np.abs(np.max(traj) - np.min(traj))/x_size
	c_i = np.abs(np.gradient(-traj, 1) - axis_ratio).argmin()
	c = traj[c_i]
	log_base_traj = np.exp2(np.log2(x)/(-traj + c))
	log_base_traj = log_base_traj[np.isfinite(log_base_traj)]
	log_base = np.median(log_base_traj[x_size//2:])
	# print(log_base)

	y = np.power(log_base, -traj) * 1e4
	a, b, c, d = np.polyfit(x, y, 3)
	x = x[-1]
	fit_vec = a*x**3 + b*x**2 + c*x + d
	fit_grad = 2*a*x + b
	y = y[-1]
	diff = (y - fit_vec)/(abs(y-fit_vec) + fit_vec)
	diff_list.append(diff)
	diff = y - fit_vec
	diff2_list.append(diff)


traj = trajectory_loss
x_size = len(traj)
x = np.arange(x_size)

axis_ratio = np.abs(np.max(traj) - np.min(traj))/x_size
c_i = np.abs(np.gradient(-traj[:10], 1) - axis_ratio).argmin()
c = traj[c_i]
log_base_traj = np.exp2(np.log2(x)/(-traj + c))
log_base = np.median(log_base_traj[x_size//2:])
# print(log_base)

y = np.power(log_base, -traj) * 1e4
a, b, c, d = np.polyfit(x, y, 3)
fit_vec = a*x**3 + b*x**2 + c*x + d
fit_grad = 2*a*x + b
diff = y - fit_vec

# plot the ployfit
ax.plot(y, label="y", color=f"C{0}")
ax.plot(fit_vec, label="fit", color=f"C{1}")
# ax.plot(diff, label="diff", color=f"C{2}")
ax.plot(diff_list, label="diff", color=f"C{3}")
# ax.plot(diff2_list, label="diff2", color=f"C{4}")


# lin_reg_window = 2
# gradient = []
# second_gradient = []
# for i in range(lin_reg_window, len(traj)):
# 	sample = traj[i-lin_reg_window:i]
# 	grad = np.gradient(sample)
# 	gradient.append(np.sum((grad)))
# 	second_gradient.append(np.sum(np.abs(np.gradient(grad))))
# ax.plot(np.arange(lin_reg_window-1, len(traj)-1),gradient, color=f"C{1}")
# ax.plot(np.arange(lin_reg_window-1, len(traj)-1),second_gradient, color=f"C{2}")

ax.grid()
ax.legend()
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.show()