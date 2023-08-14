import os
import glob
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def reorder_list(list1, list2):
    for elem in list2:
        if elem in list1:
            list1.remove(elem)
            list1.insert(0, elem)
    return list1

parser = argparse.ArgumentParser(description='Compare the runs')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--usemax', action='store_true', help='use max value instead of last 5 average')

args = parser.parse_args()

scalars_to_compare = ['i_roc', 'p_roc', 'p_pro', 'loss']


datasets = set()
exp_types = set()

for tf_event in glob.glob(args.logdir + '/**/events.out.tfevents.*', recursive=True):
	ea = event_accumulator.EventAccumulator(tf_event)
	file_path = Path(tf_event).relative_to(args.logdir)
	dataset = file_path.parent.name
	datasets.add(dataset)
	exp_type = file_path.parts[0]
	exp_types.add(exp_type)

datasets = list(sorted(datasets)) + ["average"]
exp_types = reorder_list(list(sorted(exp_types)), ["amplitude", "baseline2", "baseline"])

datasets_map = {dataset: i for i, dataset in enumerate(datasets)}
exp_types_map = {exp_type: i for i, exp_type in enumerate(exp_types)}
scalars_to_compare_map = {scalar: i for i, scalar in enumerate(scalars_to_compare)}

values = np.zeros((len(exp_types), len(datasets), len(scalars_to_compare))) - 1

for tf_event in glob.glob(args.logdir + '/**/events.out.tfevents.*', recursive=True):
	ea = event_accumulator.EventAccumulator(tf_event)
	file_path = Path(tf_event).relative_to(args.logdir)
	dataset_idx = datasets_map[file_path.parent.name]
	exp_type_idx = exp_types_map[file_path.parts[0]]

	ea.Reload() # loads events from file
	tags = ea.Tags() # print available tags

	scalars_keys = tags["scalars"]
	tensors_keys = tags["tensors"]

	for scalar_key in scalars_to_compare:
		scalars_to_compare_idx = scalars_to_compare_map[scalar_key]
		scalars = ea.Scalars(scalar_key)

		if 'loss' in scalar_key:
			comp_value = np.array([x.value for x in scalars]).min()
		else:
			if args.usemax:
				comp_value = np.array([x.value for x in scalars]).max()
			else:
				comp_value = np.array([x.value for x in scalars])[-5:].mean() # paper way

		values[exp_type_idx][dataset_idx][scalars_to_compare_idx] = comp_value

for v in values:
	mean = v[:-1].mean(axis=0)
	v[-1] = mean

# create table with colored cells for max values
n_rows = len(scalars_to_compare)
fig, axs = plt.subplots(n_rows, 1, figsize=(10, 3.55 * n_rows), dpi=300)

for ax, title in zip(axs, scalars_to_compare):
	ax.axis('off')
	ax_title = ax.set_title(title, fontsize=20)
	ax_title.set_y(1.5)
	i = scalars_to_compare_map[title]
	values_formatted = np.char.mod('%.2f', values[:,:,i])
	values_formatted = np.where(values[:,:,i] == -1, '--', values_formatted)

	# normalization
	min_v = values[:, :, i]
	min_v = np.where(min_v > 0, min_v, 99.99).min(axis=0)[None, :]
	# subtract min value from each row
	norm_v = values[:,:,i] - (min_v)
	norm_v = norm_v / (norm_v.max(axis=0)[None, :] + 1e-10)
	norm_v = 1 - norm_v if 'loss' in title else norm_v
	table = ax.table(
		cellText=values_formatted,
		rowLabels=exp_types,
		colLabels=datasets,
		cellColours=plt.cm.viridis(norm_v),
		loc='center',
		cellLoc='center',
		bbox=[0.35, 0, 0.8, 1]
	)
	table.set_fontsize(25)
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			cell = table.get_celld()[row+1, col]
			# set white text for dark cells
			if np.linalg.norm(np.array(cell.get_facecolor()), 2) < 1.2:
				cell.set_text_props(color='whitesmoke')
	for key, cell in table.get_celld().items():
		cell.set_linewidth(0.2)

ext = ".eps"
tabletype = "max" if args.usemax else "last5avg"
filename = f"table_{tabletype}"
save_dir = os.path.join(args.logdir, "comparison", f"{filename}{ext}")

i = 1
while os.path.exists(save_dir):
	save_dir = os.path.join(args.logdir, "comparison", f"{filename}_{i}{ext}")
	i += 1

plt.savefig(save_dir)

