from typing import List
from pymilvus import MilvusClient
from EmbedModel import EmbedModel 
import torch



class MilvusDAL:
    def __init__(self, uri="http://localhost:19530", max_length=512):
        self.__client = MilvusClient(
            uri=uri
        )
        self.model = EmbedModel(max_length=max_length)
    
    def insert(self, data, collection_name):
        try:
            encoded_data = self.model.encode(data)
            formatted_data = self.format_data(encoded_data, collection_name)
            res = self.__client.insert(
                collection_name=collection_name,
                data=formatted_data
            )
            print(res)
        except Exception as e:
            print(e)

    def format_data(self, data, collection_name):
        # Currently, only setup for LLM2Vec-Sheared-LLaMA
        ids = torch.arange(data.size(0))
        formatted_data = [{"id": int(id), "vector": vector.tolist()} for id, vector in zip(ids, data)]
        return formatted_data
    
    def create_collection(self, collection_name, dimension):
        try:
            res = self.__client.create_collection(
                collection_name=collection_name,
                dimension=dimension
            )
            print(res)
        except Exception as e:
            print(e)
    
    def query(self, query, collection_name, limit, metric_type="COSINE", params=None):
        if params is None:
            params = {}
        try:
            encoded_data = self.model.encode(query)
            queries = []
            for data in encoded_data.tolist():
                queries.append(data)
            res = self.__client.search(
                collection_name=collection_name,
                data=queries,
                limit=limit,
                search_params={"metric_type": metric_type, "params": params}
            )
            return res
        except Exception as e: 
            print(e)
    
    def delete_collection(self, collection_name):
        try:
            res = self.__client.drop_collection(
                collection_name=collection_name
            )
            print(res)
        except Exception as e:  
            print(e)


