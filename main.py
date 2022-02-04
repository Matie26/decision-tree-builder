import click
import pandas as pd
import decision_tree as dt
import data_preparation as dp


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', show_default=True, prompt='Path to the dataset', help='Path to the whole dataset')
@click.option('--category-column', '-c', type=int, show_default=True, prompt='Index of the column in which category is stored', help='Index of the column in which category is stored')
def prepare_data(source, category_column):
    dp.make_files(source, category_column)
    print(f'[+] Finished preparing data from {source}!')


@cli.command()
@click.option('--training-data', '-d', default='data/training.data', show_default=True, prompt='Path to the training data', help='Path to the training data')
@click.option('--category-column', '-c', type=int, show_default=True, prompt='Index of the column in which category is stored', help='Index of the column in which category is stored')
@click.option('--path-to-save', '-p', prompt='Path to save a tree', help='Path to save a tree')
@click.option('--trimming-data', '-t', default='data/trimming.data', show_default=True, prompt='Path to the trimming data', help='Path to the trimming data')
def build_tree(training_data, category_column, trimming_data, path_to_save):
    train_data = pd.read_csv(training_data, header=None)
    trim_data = pd.read_csv(trimming_data, header=None)
    tree = dt.DecisionTree(training_data=train_data,
                           col_with_category=int(category_column))
    tree = tree.trim_tree(trim_data)
    tree.save_to_file(path_to_save)
    print(f'[+] Finished building tree: {path_to_save}!')


@cli.command()
@click.option('--training-data', '-d', default='data/training.data', show_default=True, prompt='Path to the training data', help='Path to the training data')
@click.option('--category-column', '-c', type=int, show_default=True, prompt='Index of the column in which category is stored', help='Index of the column in which category is stored')
@click.option('--path-to-save', '-p', prompt='Path to save a tree', help='Path to save a tree')
def build_tree_without_trim(training_data, category_column, path_to_save):
    train_data = pd.read_csv(training_data, header=None)
    tree = dt.DecisionTree(training_data=train_data,
                           col_with_category=int(category_column))
    tree.save_to_file(path_to_save)
    print(f'[+] Finished building (w/o trim) tree: {path_to_save}!')


@cli.command()
@click.option('--path', '-p', prompt='Path to the tree file', help='Path to the tree file')
@click.option('--path-to-save', '-p', prompt='Path to save trimmed tree', help='Path to save trimmed tree')
@click.option('--trimming-data', '-t', default='data/trimming.data', show_default=True, prompt='Path to the trimming data', help='Path to the trimming data')
def trim_tree(path, path_to_save, trimming_data):
    trim_data = pd.read_csv(trimming_data, header=None)
    tree = dt.DecisionTree(path_to_file=path)
    tree = tree.trim_tree(trim_data)
    tree.save_to_file(path_to_save)
    print(
        f'[+] Finished trimming tree {path}, reuslt tree saved as {path_to_save}!')


@cli.command()
@click.option('--path', '-p', prompt='Path to the tree file', help='Path to the tree file')
def print_tree_info(path):
    tree = dt.DecisionTree(path_to_file=path)
    print(tree)


@cli.command()
@click.option('--path', '-p', prompt='Path to the tree file', help='Path to the tree file')
@click.option('--testing-data', '-t', default='data/testing.data', show_default=True, prompt='Path to the testing data', help='Path to testing data')
def test_tree_accuracy(path, testing_data):
    test_data = pd.read_csv(testing_data, header=None)
    tree = dt.DecisionTree(path_to_file=path)
    print(
        f'[+] Accuracy of tree:{path} for data:{testing_data} is:{tree.test_accuracy(test_data)}')


@cli.command()
@click.option('--tree', '-t', prompt='Path to the tree file', help='Path to the tree file')
@click.option('--data', '-d', prompt='Path to the dataset', help='Path to the dataset')
def categorize_data(tree, data):
    data_to_process = pd.read_csv(data, header=None)
    buit_tree = dt.DecisionTree(path_to_file=tree)
    result = []
    for index, row in data_to_process.iterrows():
        result.append(buit_tree.find_category(row))
    data_to_process['new_cat'] = result
    data_to_process.to_csv(f'{data}.categorized', header=False, index=False)
    print(
        f'[+] Finished categorizing data from {data} with use of {tree} tree, result data saved as {data}.categorized!')


if __name__ == '__main__':
    cli()
