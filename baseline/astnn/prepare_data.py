import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
from tree import ASTNode, BlockNode
import pickle
from javalang.ast import Node
import numpy as np

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'#node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def get_sequences(node, sequence):
    token, children = get_token(node), get_children(node)

    sequence.append(token)

    for child in children:
        get_sequences(child, sequence)

    if token in ['ForStatement', 'WhileStatement', 'DoStatement','SwitchStatement', 'IfStatement']:
        sequence.append('End')


def get_blocks(node, block_seq):
    name, children = get_token(node), get_children(node)
    logic = ['SwitchStatement','IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
    if name in ['MethodDeclaration', 'ConstructorDeclaration','ClassDeclaration', 'InterfaceDeclaration']:
        block_seq.append(BlockNode(node))
        body = node.body
        if body:
            for child in body:
                if get_token(child) not in logic and not hasattr(child, 'block'):
                    block_seq.append(BlockNode(child))
                else:
                    get_blocks(child, block_seq)
    elif name in logic:
        block_seq.append(BlockNode(node))
        for child in children[1:]:
            token = get_token(child)
            if not hasattr(node, 'block') and token not in logic+['BlockStatement']:
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
            block_seq.append(BlockNode('End'))
    elif name is 'BlockStatement' or hasattr(node, 'block'):
        block_seq.append(BlockNode(name))
        for child in children:
            if get_token(child)not in logic:
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
    else:
        for child in children:
            get_blocks(child, block_seq)

































