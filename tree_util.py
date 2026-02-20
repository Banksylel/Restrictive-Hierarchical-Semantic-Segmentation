# Utility functions for handling the structures of hierarchical trees.

import json
import csv

class node:

    def __init__(self, name):
        self.name = name
        self.children = []
        self.channel = None
        self.level = None
    

# Reads a text file describing a tree
def create_tree_from_textfile(filename):
    # non-empty set is a child node
    # empty set is a leaf node
    root = node("Universal class")
    current_depth = 0
    # opens as .json
    fd = open(filename, 'r')

    nodestack = [root]
    # There are three possibilities for each line:
    # Same indent as previous line. new node with same parent as previous line
    # More indented than previous line. new node with parent as previous line
    # Less indented than previous line. number of indents less is how many nodes need popping off the stack to find the parent
    for i, line in enumerate(fd):
        # Create new node with the name taken from the text file after stripping tabs
        newnode = node(line.strip())
        if line.count('\t') == current_depth:
            # Make new node child of node currently at top of stack
            nodestack[-1].children.append(newnode)
            # Make a copy in case next line is indented more
            prevnode = newnode
        elif line.count('\t') == current_depth + 1:
            # We have just indented one so the previous line is the parent of this node

            # Add node from previous line to top of stack
            nodestack.append(prevnode)

            # Make new node child of node currently at top of stack
            nodestack[-1].children.append(newnode)

            current_depth += 1

            prevnode = newnode
        elif line.count('\t') < current_depth:
            # Indentation has reduced
            new_depth = line.count('\t')
            # For each reduction in indentation, pop one node off the stack
            while current_depth > new_depth:
                nodestack.pop()
                current_depth -= 1
            # Make new node child of node currently at top of stack
            nodestack[-1].children.append(newnode)

            prevnode = newnode
        else:
            raise RuntimeError("Indentation can only increase by one")
    fd.close()
    return root


# Adds a channel to a specific node
def add_channels(node, channel):
    if len(node.children) == 0:
        node.channel = channel
        return channel + 1
    else:
        for child in node.children:
            channel = add_channels(child, channel)
        return channel


def update_channels(node, class_lookup):  # Created to map channels in labels to those in tree
    if len(node.children) == 0:
        node.channel = class_lookup[node.channel]
        return
    else:
        for child in node.children:
            update_channels(child, class_lookup)


# Needs the depth given as argument. Root not allocated level
# Could get rid of need for depth argument
def add_levels(node, depth):
    # depth = find_depth(node) - 1
    # print(depth)

    if len(node.children) == 0:
        node.level = depth - 1
    else:
        for child in node.children:
            if len(child.children) == 0:
                child.level = depth - 1
            else:
                child.level = depth - 1
                add_levels(child, depth - 1)


# Gets the names of all leaf nodes of the tree (predicted classes, as parent nodes are technically not classes)
def getLeafClasses(node, my_list):
    if len(node.children) == 0:
        my_list.append(node.channel)
        return my_list
    else:
        for child in node.children:
            getLeafClasses(child, my_list)
        return my_list


# Returns the depth of a tree at a given node
def find_depth(node):
    if len(node.children) == 0:
        return 0
    else:
        maxlen = 0
        for child in node.children:
            maxlen = max(maxlen, find_depth(child))
        return maxlen + 1


def getLossLevelList(root, level, myList):
    for child in root.children:
        if len(child.children) == 0 or child.level == level:
            myList.append(getLeafClasses(child, []))
        else:
            getLossLevelList(child, level, myList)


def getTreeList(node):
    depth = find_depth(node)
    main_list = []
    for level in range(depth):
        level_list = []
        getLossLevelList(node, level, level_list)
        main_list.append(level_list)
    return main_list
