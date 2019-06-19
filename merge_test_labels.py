"""Script for merging the test data with ground truth labels."""
import os
import csv
import glovar


if __name__ == '__main__':
    # make sure the data is there
    test_only_data_path = os.path.join(glovar.ARCT_DIR, 'test-only-data.txt')
    truth_data_path = os.path.join(glovar.ARCT_DIR, 'truth.txt')
    test_data_path = os.path.join(glovar.ARCT_DIR, 'test-full.txt')
    if not os.path.exists(test_only_data_path):
        raise ValueError('Missing text-only-data.txt in data dir. '
                         'Run prepare.sh.')
    if not os.path.exists(truth_data_path):
        raise ValueError('Missing truth.txt in data dir. Run prepare.sh.')

    # grab the labels from the truth file
    with open(truth_data_path, 'r') as f:
        lines = f.readlines()
        # there are some comment lines we don't want
        # also strip off the endlines
        lines = [l.strip() for l in lines if not l.startswith('#')]
        ids = []
        labels = []
        # there is one line that is split with three spaces
        for line in lines:
            if '\t' in line:
                _id, label = line.split('\t')
            else:  # this is the only other case
                _id, label = line.split('   ')
            ids.append(_id)
            labels.append(int(label))
        label_dict = dict(zip(ids, labels))

    # load the test data
    rows = []
    with open(test_only_data_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            rows.append(row)

    # merge 'em
    rows[0].append('correctLabelW0orW1')
    for row in rows[1:]:
        _id = row[0]
        label = label_dict[_id]
        row.append(label)

    # save output
    with open(test_data_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in rows:
            writer.writerow(row)
