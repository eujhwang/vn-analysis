import torch


class ToDense(object):
    r"""Converts a sparse adjacency matrix to a dense adjacency matrix with
    shape :obj:`[num_nodes, num_nodes, *]`.

    Args:
        num_nodes (int): The number of nodes. If set to :obj:`None`, the number
            of nodes will get automatically inferred. (default: :obj:`None`)
    """
    def __init__(self, num_nodes=None, remove_edge_index: bool = False):  # VT added the latter!
        self.num_nodes = num_nodes
        self.remove_edge_index = remove_edge_index

    def __call__(self, data):
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes
        if self.num_nodes is None:
            num_nodes = orig_num_nodes
        else:
            assert orig_num_nodes <= self.num_nodes
            num_nodes = self.num_nodes

        if data.edge_attr is None:
            edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
        adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)
        data.adj = adj.to_dense()
        if self.remove_edge_index:
            data.edge_index = None
            data.edge_attr = None

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        if data.y is not None and (data.y.size(0) == orig_num_nodes):
            size = [num_nodes - data.y.size(0)] + list(data.y.size())[1:]
            data.y = torch.cat([data.y, data.y.new_zeros(size)], dim=0)

        return data

    def __repr__(self):
        if self.num_nodes is None:
            return '{}()'.format(self.__class__.__name__)
        else:
            return '{}(num_nodes={})'.format(self.__class__.__name__,
                                             self.num_nodes)
