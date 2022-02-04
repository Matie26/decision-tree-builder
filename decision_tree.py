import sys
import math
import pickle
import copy
import pandas as pd


class DecisionTree:
    def __init__(self, training_data=None, col_with_category=None, path_to_file=None):
        self.leaf_counter = 0
        self.current_node_id = 0
        if path_to_file:
            infile = open(path_to_file, 'rb')
            self.__dict__.update(pickle.load(infile).__dict__)
            infile.close()
        else:
            self.training_data = training_data
            self.col_with_category = col_with_category
            print('Building tree ...')
            self.root = self.__build_tree(training_data)

    def __str__(self):
        map = self.__traverse(self.root)
        leafes = 0
        for row in map:
            for element in row:
                if type(element).__name__ == 'Leaf':
                    leafes += 1
        return f'Decision tree built from {len(self.training_data)} elements, leafs: {leafes}'

    def trim_tree(self, data):
        tree = self
        processed_levels = 0
        while True:
            print('! Trimmed !')
            temp, changed, processed_levels = tree.__trim_one_node(
                data, processed_levels=processed_levels)
            if not changed:
                break
            tree = temp
        return tree

    def test_accuracy(self, test_data):
        pos = 0
        for index, row in test_data.iterrows():
            if self.find_category(row) == row[self.col_with_category]:
                pos += 1
        return pos/len(test_data)

    def find_category(self, row):
        current_node = self.root
        while(type(current_node).__name__ != 'Leaf'):
            current_node = current_node.get_child_node(
                more_or_equal=row[current_node.attribute] >= current_node.value)
        return current_node.category

    def save_to_file(self, path):
        outfile = open(path, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def __build_tree(self, data):
        print(f'build tree from {len(data)} elements')
        self.current_node_id += 1
        if self.__stop_criteria(data):
            tree = Leaf(self.current_node_id, category=self.__get_category(
                data), n_of_elements=len(data))
            self.leaf_counter += 1
        else:
            tests = self.__generate_test_pool(data)
            results = ([], [])
            while len(results[0]) == 0 or len(results[1]) == 0:
                choosen_test = self.__choose_test(tests)
                results = self.__split_by_attribute(data, choosen_test)
            tree = Node(self.current_node_id, attribute=choosen_test[0], value=choosen_test[1])
            tree.set_child_node(more_or_equal=True, node=self.__build_tree(results[0]))
            tree.set_child_node(more_or_equal=False, node=self.__build_tree(results[1]))
        return tree

    def __test_quality(self, data, attr, split_point):
        subsets = self.__split_by_attribute(data, (attr, split_point, 0))
        sum0 = len(subsets[1])
        sum1 = len(subsets[0])
        suk0 = subsets[1][self.col_with_category].sum()
        suk1 = subsets[0][self.col_with_category].sum()

        if (sum0 == 0):
            E_0 = 0
        elif (suk0 == 0 or suk0 == sum0):
            E_0 = 0
        else:
            E_0 = -(suk0/sum0)*math.log10(suk0/sum0) - \
                ((sum0-suk0)/sum0)*math.log10((sum0-suk0)/sum0)
        if (sum1 == 0):
            E_1 = 0
        elif (suk1 == 0 or suk1 == sum1):
            E_1 = 0
        else:
            E_1 = -(suk1/sum1)*math.log10(suk1/sum1) - \
                ((sum1-suk1)/sum1)*math.log10((sum1-suk1)/sum1)
        E_w = sum0/len(data) * E_0 + sum1/len(data) * E_1
        return(E_w)

    def __stop_criteria(self, data):
        if len(data) == 0 or data[self.col_with_category].nunique() == 1:
            return True
        return False

    def __split_by_attribute(self, data, test):

        subset1 = data[data[test[0]] >= test[1]]
        subset2 = data[data[test[0]] < test[1]]
        return (subset1, subset2)

    def __get_category(self, data):
        if len(data) == 0:
            return -1
        sum = data[self.col_with_category].sum()
        if sum/len(data) >= 0.5:
            return 1
        else:
            return 0

    def __generate_test(self, data, attr, min, max):
        mid = (min + max) / 2
        entropy1 = self.__test_quality(data, attr, mid)
        entropy2 = self.__test_quality(data, attr, (min+mid)/2)
        entropy3 = self.__test_quality(data, attr, (max+mid)/2)

        if entropy1 <= entropy2 and entropy1 <= entropy3:
            split_point = mid
            result_entropy = entropy1
        elif entropy2 <= entropy3:
            split_point, result_entropy = self.__generate_test(
                data, attr, min, mid)
        else:
            split_point, result_entropy = self.__generate_test(
                data, attr, mid, max)
        return split_point, result_entropy

    def __generate_test_pool(self, data):
        result = []
        for c in data.columns:
            if c != self.col_with_category:
                split_point, result_entropy = self.__generate_test(
                    data, c, data[c].min(), data[c].max())
                result.append((c, split_point, result_entropy))
        return result

    def __choose_test(self, tests):
        tests_df = pd.DataFrame(
            tests, columns=['index', 'split_point', 'entropy'])
        weights = tests_df.loc[:, 'entropy'].values.tolist()
        for i in range(len(weights)):
            if weights[i] == 0:
                weights[i] = sys.maxsize * 2 + 1
            else:
                weights[i] = 1 / weights[i]
        chosen_test_df = tests_df.sample(n=1, weights=weights)
        chosen_test = chosen_test_df.iloc[0].to_numpy()
        return chosen_test

    def __traverse(self, rootnode, id=None):
        tree_map = []
        thislevel = [rootnode]
        while thislevel:
            nextlevel = list()
            for n in thislevel:
                if n.node_id == id:
                    return n
                if type(n).__name__ != 'Leaf':
                    nextlevel.append(n.get_child_node(True))
                    nextlevel.append(n.get_child_node(False))
            tree_map.append(thislevel)
            thislevel = nextlevel
        return tree_map

    def __tree_to_leaf(self, rootnode):
        categories = {}
        thislevel = [rootnode]
        while thislevel:
            nextlevel = list()
            for n in thislevel:
                if type(n).__name__ != 'Leaf':
                    nextlevel.append(n.get_child_node(True))
                    nextlevel.append(n.get_child_node(False))
                else:
                    if n.category not in categories:
                        categories[n.category] = 0
                    categories[n.category] += n.n_of_elements
            thislevel = nextlevel
        winning_cat = max(categories, key=categories.get)
        leaf = Leaf(-1, category=winning_cat,
                    n_of_elements=sum(categories.values()))
        return leaf

    def __trim_one_node(self, data, processed_levels):
        trimed_tree = copy.deepcopy(self)
        trimed_tree_map = self.__traverse(trimed_tree.root)
        row_number = 0
        for row in trimed_tree_map:
            row_number += 1
            if row_number < processed_levels:
                continue
            for node in row:
                if type(node).__name__ != 'Leaf':
                    print(f'Working on {type(node).__name__} {node.node_id}')
                    for side in [True, False]:
                        temp_node = node.get_child_node(side)
                        if type(temp_node).__name__ == 'Leaf':
                            continue
                        node.set_child_node(side, self.__tree_to_leaf(
                            node.get_child_node(side)))
                        old_acc = self.test_accuracy(data)
                        new_acc = trimed_tree.test_accuracy(data)
                        print(
                            f'\tFor {side} branch:\t old={old_acc} new={new_acc} better? {new_acc>=old_acc}')
                        if new_acc >= old_acc:
                            print(f'Trimming {side} side')
                            return trimed_tree, True, processed_levels
                        node.set_child_node(side, temp_node)
            processed_levels += 1
        return None, False, 0


class Node():
    def __init__(self, node_id, attribute=None, value=None):
        self.node_id = node_id
        self.attribute = attribute
        self.value = value
        self.__more_eq_node = None
        self.__less_node = None

    def set_child_node(self, more_or_equal, node):
        if more_or_equal:
            self.__more_eq_node = node
        else:
            self.__less_node = node

    def get_child_node(self, more_or_equal):
        if more_or_equal:
            return self.__more_eq_node
        else:
            return self.__less_node


class Leaf():
    def __init__(self, node_id, category=None, n_of_elements=0):
        self.node_id = node_id
        self.category = category
        self.n_of_elements = n_of_elements
