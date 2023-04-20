import glob
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser(description='Compare the runs')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')

args = parser.parse_args()

scalars_to_compare = ['i_roc', 'p_roc', 'p_pro']


datasets = set()
exp_types = set()

for tf_event in glob.glob(args.logdir + '/**/events.out.tfevents.*', recursive=True):
	ea = event_accumulator.EventAccumulator(tf_event)
	file_path = Path(tf_event)
	dataset = file_path.parent.name
	datasets.add(dataset)
	exp_type = file_path.parts[1]
	exp_types.add(exp_type)

datasets = list(sorted(datasets)) + ["average"]
exp_types = list(sorted(exp_types))

datasets_map = {dataset: i for i, dataset in enumerate(datasets)}
exp_types_map = {exp_type: i for i, exp_type in enumerate(exp_types)}
scalars_to_compare_map = {scalar: i for i, scalar in enumerate(scalars_to_compare)}

values = np.zeros((len(exp_types), len(datasets), len(scalars_to_compare))) - 1

for tf_event in glob.glob(args.logdir + '/**/events.out.tfevents.*', recursive=True):
	ea = event_accumulator.EventAccumulator(tf_event)
	file_path = Path(tf_event)
	dataset_idx = datasets_map[file_path.parent.name]
	exp_type_idx = exp_types_map[file_path.parts[1]]

	ea.Reload() # loads events from file
	tags = ea.Tags() # print available tags

	scalars_keys = tags["scalars"]
	tensors_keys = tags["tensors"]

	for scalar_key in scalars_to_compare:
		scalars_to_compare_idx = scalars_to_compare_map[scalar_key]
		scalars = ea.Scalars(scalar_key)
		max = np.array([x.value for x in scalars]).max()
		values[exp_type_idx][dataset_idx][scalars_to_compare_idx] = max

for v in values:
	mean = v[:-1].mean(axis=0)
	v[-1] = mean

# create table with colored cells for max values
n_rows = len(scalars_to_compare)
fig, axs = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows))
for ax, title in zip(axs, scalars_to_compare):
	ax.axis('off')
	ax.set_title(title, fontsize=20)
	i = scalars_to_compare_map[title]
	values_formatted = np.char.mod('%.2f', values[:,:,i])
	values_formatted = np.where(values[:,:,i] == -1, '--', values_formatted)

	# normalization
	min_v = values[:, :, i]
	min_v = np.where(min_v > 0, min_v, 99.99).min(axis=0)[None, :]
	# subtract min value from each row
	norm_v = values[:,:,i] - (min_v)
	norm_v = norm_v / (norm_v.max(axis=0)[None, :] + 1e-10)
	ax.table(
		cellText=values_formatted,
		rowLabels=exp_types,
		colLabels=datasets,
		cellColours=plt.cm.Greens(norm_v),
		loc='center',
		fontsize=20
	)

plt.savefig("comparison.png")

