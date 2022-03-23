import json
import networkx as nx
import matplotlib.pyplot as plt
import operator
import pdb

filename = "tmp_dblp_coauthorship.json"

def load_json():
    fobj = open(filename)
    file = json.load(fobj)
    return file

def load_2005(jsonfile):
    list_2005 = []
    for lst in jsonfile:
        if lst[2] == 2005:
            list_2005.append(lst)
    #print(list_2005)
    return list_2005

def load_2006(jsonfile):
    list_2006 = []
    for lst in jsonfile:
        if lst[2] == 2006:
            list_2006.append(lst)
    #print(list_2006)
    return list_2006

def unweighted_graph_2005(list_2005):
    G = nx.Graph()
    G.add_nodes_from(lst[0] for lst in list_2005)
    G.add_nodes_from(lst[1] for lst in list_2005)

    for lst in list_2005:       
        G.add_edge(lst[0], lst[1])
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    print(f"Example of unweighted 2005 graph nodes are {list(G0.nodes)[:5]}")
    print(f"The 2005 unweighted graph has {len(G0.nodes)} nodes")
    print("---------------------------------------------------")
    print(f"Example of unweighted 2005 graph edges are {list(G0.edges)[:5]}")
    print(f"The 2005 unweighted graph has {len(G0.edges)} edges")
    
    return G0

def unweighted_graph_2006(list_2006):
    G = nx.Graph()
    G.add_nodes_from(lst[0] for lst in list_2006)
    G.add_nodes_from(lst[1] for lst in list_2006)

    for lst in list_2006:       
        G.add_edge(lst[0], lst[1])
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    print(f"Example of unweighted 2006 graph nodes are {list(G0.nodes)[:5]}")
    print(f"The 2006 unweighted graph has {len(G0.nodes)} nodes")
    print("---------------------------------------------------")
    print(f"Example of unweighted 2006 graph edges are {list(G0.edges)[:5]}")
    print(f"The 2006 unweighted graph has {len(G0.edges)} edges")
    
    return G0

def weighted_graph_2005(list_2005):
    G = nx.Graph()
    G.add_nodes_from(lst[0] for lst in list_2005)
    G.add_nodes_from(lst[1] for lst in list_2005)

    weights = {}

    for lst in list_2005:
        if (lst[0], lst[1]) not in weights:
            weights[(lst[0],lst[1])] = 1
        else:
            weights[(lst[0],lst[1])] += 1
    for lst in list_2005:
        G.add_edge(lst[0], lst[1], weight = weights[(lst[0],lst[1])])
    
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    # nx.set_edge_attributes(G, values = weights, name = 'weights')
    print(f"Example of weighted 2005 graph nodes are {list(G0.nodes)[:5]}")
    print(f"The 2005 weighted graph has {len(G0.nodes)} nodes")
    print("---------------------------------------------------")
    print(f"Example of weighted 2005 graph edges are {list(G0.edges)[:5]}")
    print(f"The 2005 weighted graph has {len(G0.edges)} edges")
    print("---------------------------------------------------")
    
    return G0
    
    # test weights
    """high_corr = [(u,v,w) for (u,v,w) in G.edges(data = True) if w["weight"] > 1]
    print(high_corr)"""


# Part B
# PageRanks
def sorted_pageranks(graph):
    top_50_authors = {}
    pagerank_dict = nx.pagerank(graph)
    for k,v in sorted(pagerank_dict.items(), key=lambda x: x[1], reverse = True)[:50]:
        top_50_authors[k] = format((v * 100),'.3f')
    print("------------Singer Author Top 50-------------")
    print(top_50_authors)
    return top_50_authors

# Edge Betweenness Scores
def sorted_edge_betweenness(graph):
    top_50_coauthors = {}
    edge_between = nx.edge_betweenness_centrality(graph)
    # pdb.set_trace()
    for k,v in sorted(edge_between.items(), key = lambda x:x[1],reverse = True)[:50]:
        top_50_coauthors[k] = v
    print("------------Co-Author Top 50-------------")
    print(top_50_coauthors)
    return top_50_coauthors

# Part C
# i
def dblp2005_core(dblp2005):
    core_2005 = dblp2005.copy()
    to_be_removed = [node for node in dblp2005.nodes if dblp2005.degree(node) < 3]
    for node in to_be_removed:
        core_2005.remove_node(node)
    print(f"The core 2005 unweighted graph has {len(core_2005.nodes)} nodes")
    print("---------------------------------------------------")
    print(f"The core 2005 unweighted graph has {len(core_2005.edges)} edges")
    print("---------------------------------------------------")
    return core_2005

# ii
def dblp2006_core(dblp2006):
    core_2006 = dblp2006.copy()
    to_be_removed = [node for node in dblp2006.nodes if dblp2006.degree(node) < 3]
    for node in to_be_removed:
        core_2006.remove_node(node)
    print(f"The core 2006 unweighted graph has {len(core_2006.nodes)} nodes")
    print("---------------------------------------------------")
    print(f"The core 2006 unweighted graph has {len(core_2006.edges)} edges")
    print("---------------------------------------------------")
    return core_2006

# iii
# helper function
def friend_set(graph, node):
    return set(graph.neighbors(node))
# friend of friend set
def friend_of_friend_set(graph, friends):
    fof_set = set()

    for f in friends:
        fof_set.update(graph.neighbors(f))
    return fof_set
# main fof function
def find_fof_core(graph, node):
    friends = friend_set(graph, node)
    friend_of_friend = friend_of_friend_set(graph,friends)
    return friend_of_friend

# iv
def unique_2006_edges(G2005_edge,G2006_edge):
    return G2006_edge - G2005_edge


# v
# a) RD

# b) CN
def common_neighbors_prediction(graph):
    list_graph_edges = list(graph.edges)
    cn_dict = dict()
    preds = nx.common_neighbor_centrality(graph, list_graph_edges)
    pdb.set_trace()
    for u,v,p in preds:
        cn_dict[(u,v)] = p
    sorted_keys = sorted(cn_dict, key=cn_dict.get)
    sorted_dict = dict()
    for w in sorted_keys:
        sorted_dict[w] = cn_dict[(u,v)]
    return sorted_dict


# c) JC
def jaccard_coe(graph):
    return(list(nx.jaccard_coefficient(graph)))

# d) PA

def pref_attachment(graph):
    list_graph_edges = list(graph.edges)
    preds = nx.preferential_attachment(graph, list_graph_edges)
    pa_dict = dict()
    for u,v,p in preds:
        pa_dict[(u,v)] = p
    sorted_keys = sorted(pa_dict, key=pa_dict.get)
    sorted_dict = dict()
    for w in sorted_keys:
        sorted_dict[w] = pa_dict[(u,v)]
    return sorted_dict

# e) AA

def adamic(graph):
    list_graph_edges = list(graph.edges)
    preds = nx.adamic_adar_index(graph,list_graph_edges)
    aa_dict = dict()
    for u,v,p in preds:
        aa_dict[(u,v)] = p
    sorted_keys = sorted(aa_dict, key=aa_dict.get)
    sorted_dict = dict()
    for w in sorted_keys:
        sorted_dict[w] = aa_dict[(u,v)]
    return sorted_dict


def main():
    jsonfile = load_json()

    # 2005
    list_2005 = load_2005(jsonfile)
    u_graph_2005 = unweighted_graph_2005(list_2005)

    # Node & Edge Importance
    u_2005_pr = sorted_pageranks(u_graph_2005)

    # edge_betweenness taking too long
    # sorted_edge_betweenness(u_graph_2005)

    # CORE 2005
    dblp2005core_graph = dblp2005_core(u_graph_2005)


    # 2006
    list_2006 = load_2006(jsonfile)
    u_graph_2006 = unweighted_graph_2006(list_2006)
    
    # Node & Edge Importance
    sorted_pageranks(u_graph_2006)

    # edge_betweenness taking too long
    # sorted_edge_betweenness(u_graph_2006)
    
    # CORE 2006
    dblp2006core_graph = dblp2006_core(u_graph_2006)


    # 2005 weighted
    list_2005 = load_2005(jsonfile)
    w_graph_2005 = weighted_graph_2005(list_2005)
    # Node & Edge Importance
    sorted_pageranks(w_graph_2005)
    # edge_betweenness taking too long
    # sorted_edge_betweenness(w_graph_2005)


    # fof
    fof_res = list()
    for node in dblp2005core_graph.nodes:
        fof_res.append(find_fof_core(dblp2005core_graph, node))
    print(f"fof_res is {fof_res}")
    
    # Unique 2006Core edges:
    edge_2005_core_set = set(dblp2005core_graph.edges)
    edge_2006_core_set = set(dblp2006core_graph.edges)
    unique_2006 = unique_2006_edges(edge_2005_core_set, edge_2006_core_set)
    print(unique_2006)

    # cn
    # cn_dict = common_neighbors_prediction(dblp2005core_graph)
    # jc
    # jc_list = jaccard_coe(dblp2005core_graph)
    # pa
    # pa_dict = pref_attachment(dblp2005core_graph)
    # aa
    # aa_dict = adamic(dblp2005core_graph)

    k = [10,20,50,100]




if __name__ == "__main__":
    main()