import os
import random
import pandas as pd
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
import torch
from anytree import AnyNode, RenderTree
#import treelib
from anytree import find
from edge_index import edges
from selected_nodes import selected_nodes
import sys
sys.setrecursionlimit(9999)

def get_token(node):
    token = ''
    #print(isinstance(node, Node))
    #print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):
    #print(root)
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
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    if token.find('/*')==-1:  # 去除多行注释
        # print(node.__class__, token)
        sequence.append(token)

        for child in children:
            get_sequence(child, sequence)

def getnodes(node,nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child,nodelist)

class Queue():
    def __init__(self):
        self.__list = list()

    def isEmpty(self):
        return self.__list == []

    def push(self, data):
        self.__list.append(data)

    def pop(self):
        if self.isEmpty():
            return False
        return self.__list.pop(0)
def traverse(node,index):
    queue = Queue()
    queue.push(node)
    result = []
    while not queue.isEmpty():
        node = queue.pop()
        result.append(get_token(node))
        result.append(index)
        index+=1
        for (child_name, child) in node.children():
            #print(get_token(child),index)
            queue.push(child)
    return result

# 原始的，未简化
def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if token.find('/*') != -1:
        return
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)
def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)
def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    if not token:
        return
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)
def getedge_nextsib(node,vocabdict,src,tgt,edgetype):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append([1])
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append([edges['Prevsib']])
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt,edgetype)
def getedge_flow(node,vocabdict,src,tgt,edgetype,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['While']])
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['For']])
            '''if len(node.children[1].children)!=0:
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[0].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[-1].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopend'])
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[-1].id)
                edgetype.append(edges['For_loopend'])'''
    #if token=='ForControl':
        #print(token,len(node.children))
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['If']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['If']])
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([edges['Ifelse']])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append([edges['Ifelse']])
    for child in node.children:
        getedge_flow(child,vocabdict,src,tgt,edgetype,ifedge,whileedge,foredge)
def getedge_nextstmt(node,vocabdict,src,tgt,edgetype):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            edgetype.append([edges['Nextstmt']])
            src.append(node.children[i+1].id)
            tgt.append(node.children[i].id)
            edgetype.append([edges['Prevstmt']])
    for child in node.children:
        getedge_nextstmt(child,vocabdict,src,tgt,edgetype)
def getedge_nexttoken(node,vocabdict,src,tgt,edgetype,tokenlist):
    def gettokenlist(node,vocabdict,edgetype,tokenlist):
        token=node.token
        if len(node.children)==0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child,vocabdict,edgetype,tokenlist)
    gettokenlist(node,vocabdict,edgetype,tokenlist)
    for i in range(len(tokenlist)-1):
            src.append(tokenlist[i])
            tgt.append(tokenlist[i+1])
            edgetype.append([edges['Nexttoken']])
            src.append(tokenlist[i+1])
            tgt.append(tokenlist[i])
            edgetype.append([edges['Prevtoken']])
def getedge_nextuse(node,vocabdict,src,tgt,edgetype,variabledict):
    def getvariables(node,vocabdict,edgetype,variabledict):
        token=node.token
        if token=='MemberReference':
            if node.children:
                for child in node.children:
                    if child.token==node.data.member:
                        variable=child.token
                        variablenode=child
                        if not variabledict.__contains__(variable):
                            variabledict[variable]=[variablenode.id]
                        else:
                            variabledict[variable].append(variablenode.id)      
        for child in node.children:
            getvariables(child,vocabdict,edgetype,variabledict)
    getvariables(node,vocabdict,edgetype,variabledict)
    # print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
                src.append(variabledict[v][i])
                tgt.append(variabledict[v][i+1])
                edgetype.append([edges['Nextuse']])
                src.append(variabledict[v][i+1])
                tgt.append(variabledict[v][i])
                edgetype.append([edges['Prevuse']])

# 简化AST
# 1.深度优先遍历得到该节点的所有的叶节点
def get_leaf(root,source_code):
    token,children=get_token(root),get_child(root)
    # if token=="PackageDeclaration" or token=="Import" or token=="CompilationUnit":
    #     return
    num=len(children)
    if num==0:
        source_code.append(token)
    for child in children:
        get_leaf(child,source_code)
# 2.
def createsimplifytree(root,node,alltokens,node_list,source_code,selected_nodes,parent=None):
    token, children=get_token(node), get_child(node)

    if token in source_code or token in selected_nodes or (parent is not None and parent.token=="WhileStatement"):
        id=len(node_list)
        alltokens.append(token)
        if id==0:
            root.token=token
            root.data=node
            parent=root
            node_list.append(root)
        else:
            newnode=AnyNode(id=id,token=token,data=node,parent=parent)
            parent=newnode
            node_list.append(newnode)

    for child in children:
        createsimplifytree(root,child,alltokens,node_list,source_code,selected_nodes,parent)

def createast(project):
    astdict={}
    labeldict={}
    dirname = 'data/'
    path = dirname+'programs_'+project+'.pkl'
    df = pd.read_pickle(path)  # Index(['id', 'code', 'label'], dtype='object')
    for index, row in df.iterrows():
        programtext = row['code']
        programast=javalang.parse.parse(programtext)
        astdict[row['id']]=programast
        labeldict[row['id']]=row['label']

    #简化AST
    alltokens=[]
    delpath=[]
    for path,tree in astdict.items():
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        sourcecode=[]
        children=get_child(tree)
        for child in children:
            token=get_token(child)
            if token in ["ClassDeclaration","MethodDeclaration"]:
                tree=child
                break
        else:
            delpath.append(path)
            tree=None
        if not tree:
            continue
        get_leaf(tree,sourcecode)
        sourcecode=list(set(sourcecode))
        sourcecode=[x for x in sourcecode if x.find("/*")==-1 and x.find("//")==-1]
        createsimplifytree(newtree,tree,alltokens,nodelist,sourcecode,selected_nodes)
        astdict[path]=newtree

    # 删除没有简化的AST
    if delpath:
        for path in delpath:
            del astdict[path]

    #统计节点数
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount=0
    switchcount=0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    # print(ifcount,whilecount,forcount,blockcount,docount,switchcount)
    # print('allnodes ',len(alltokens))
    alltokens=list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    # print(vocabdict)
    return astdict,vocabsize,vocabdict,labeldict

def createseparategraph(astdict,vocablen,vocabdict,device,mode='astonly',simplify=False,nextsib=False,ifedge=False,whileedge=False,foredge=False,blockedge=False,nexttoken=False,nextuse=False):
    pathlist=[]
    treelist=[]
    # print('simplify ',simplify)
    # print('nextsib ',nextsib)
    # print('ifedge ',ifedge)
    # print('whileedge ',whileedge)
    # print('foredge ',foredge)
    # print('blockedge ',blockedge)
    # print('nexttoken', nexttoken)
    # print('nextuse ',nextuse)
    # print(len(astdict))
    newastdict={}
    for path,newtree in astdict.items():        
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        if mode=='astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
            if not x or not edgesrc or not edgetgt:
                continue
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)
            if not x or not edgesrc or not edgetgt:
                continue
            if nextsib==True:
                getedge_nextsib(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            getedge_flow(newtree,vocabdict,edgesrc,edgetgt,edge_attr,ifedge,whileedge,foredge)
            if blockedge==True:
                getedge_nextstmt(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            tokenlist=[]
            if nexttoken==True:
                getedge_nexttoken(newtree,vocabdict,edgesrc,edgetgt,edge_attr,tokenlist)
            variabledict={}
            if nextuse==True:
                getedge_nextuse(newtree,vocabdict,edgesrc,edgetgt,edge_attr,variabledict)
        edge_index=[edgesrc, edgetgt]
        astlength=len(x)
        # if astlength<=1:
        #     continue
        pathlist.append(path)
        treelist.append([[x,edge_index,edge_attr],astlength])
        astdict[path]=[[x,edge_index,edge_attr],astlength]   # path is the id of ast
        # newastdict[path]=[[x,edge_index,edge_attr],astlength] 
    return astdict
def createggnndata(treedict,labeldict,device):    

    ids = list(treedict.keys())
    random.shuffle(ids)
    n = len(ids)
    ratio = n//10
    train_index = ids[:ratio*8]
    # valid_index = ids[ratio*8:ratio*9]
    # test_index = ids[ratio*9:]
    test_index = ids[ratio*8:]

    print('train data')
    traindata=createdata(treedict,train_index,labeldict,device=device)
    # print('valid data')
    # validdata=createdata(treedict,valid_index,labeldict,device=device)
    print('test data')
    testdata=createdata(treedict,test_index,labeldict,device=device)
    return traindata, testdata #, validdata
def createdata(treedict,indexlist,labeldict,device):
    datalist=[]
    for index in indexlist:
        label=labeldict[index]
        data = treedict[index]
        x,edge_index,edge_attr,ast1length=data[0][0],data[0][1],data[0][2],data[1]
        if edge_attr==[]:
            edge_attr = None
        data = [[x, edge_index, edge_attr], label]
        datalist.append(data)
    return datalist
    

if __name__ == '__main__':
    # astdict, vocabsize, vocabdict=createast()
    # treedict=createseparategraph(astdict, vocabsize, vocabdict,device='cpu',mode='else',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--dataset", default='gcj')
    parser.add_argument("--graphmode", default='astonly')  # astandnext
    parser.add_argument("--nextsib", default=False)
    parser.add_argument("--ifedge", default=False)
    parser.add_argument("--whileedge", default=False)
    parser.add_argument("--foredge", default=False)
    parser.add_argument("--blockedge", default=False)
    parser.add_argument("--nexttoken", default=False)
    parser.add_argument("--nextuse", default=False)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--num_layers", default=4)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--threshold", default=0)
    args = parser.parse_args()
    
    device=torch.device('cuda:0')
    astdict, vocablen, vocabdict, labeldict=createast('ambari_upsample37')#ambari_upsample37
    treedict=createseparategraph(astdict, vocablen, vocabdict,device,mode=args.graphmode,nextsib=args.nextsib,ifedge=args.ifedge,whileedge=args.whileedge,foredge=args.foredge,blockedge=args.blockedge,nexttoken=args.nexttoken,nextuse=args.nextuse)
    traindata,testdata=createggnndata(treedict,labeldict,device)
    print('test ok')
