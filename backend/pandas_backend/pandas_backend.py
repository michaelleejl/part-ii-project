import pandas as pd

from backend.backend import Backend
from backend.pandas_backend.helpers import check_columns_match, copy_data, get_cols_of_node, determine_cardinality
from backend.pandas_backend.relation import Relation, DataRelation, FunctionRelation


class PandasBackend(Backend):

    def __init__(self):
        self.node_data = {}
        self.edge_data = {}

    def map_node_to_domain(self, node, domain: pd.DataFrame) -> None:
        extended_domain = copy_data(domain).rename({k: f"{node.get_key()}_{k}" for k in domain.columns}, axis=1)
        if node not in self.node_data.keys():
            self.node_data[node] = extended_domain
            return None
        else:
            old_domain = self.node_data[node]
            check_columns_match(old_domain, extended_domain)
            new_data = pd.concat([old_domain, extended_domain]).reset_index().drop_duplicates()
            self.node_data[node] = new_data

    def map_edge_to_relation(self, edge, relation: Relation):
        if type(relation) == DataRelation:
            print("ok")
            if edge not in self.edge_data.keys():
                self.edge_data[edge] = relation
            else:
                assert type(self.edge_data[edge]) == DataRelation
                self.edge_data[edge].update_relation(relation)

    def get_cardinality(self, edge, start):
        mapping = self.edge_data[edge]
        end = edge.to_node if start == edge.from_node else edge.from_node

        if type(mapping) == DataRelation:
            key_cols = get_cols_of_node(mapping.data, end)
            val_cols = get_cols_of_node(mapping.data, start)
            return determine_cardinality(mapping.data, key_cols, val_cols)

    def execute_query(self, query):
        pass