from neomodel import  StructuredNode, StringProperty,UniqueIdProperty
from src.db.helpers.graph import GraphDB

graph = GraphDB()

class DocumentChunks(StructuredNode):
    uid = UniqueIdProperty()
    data = StringProperty(unique_index=True)

    def to_json(self):
        return {
            "uid": self.uid,
            "data": self.data
        }

class Machines(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    type = StringProperty(unique_index=True, required=True)
    manufacturer = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)

class Documents(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    chunks = graph.create_relationship_to(DocumentChunks,'HAS_MANY')

    def to_json(self):
        return {
            "uid": self.uid,
            "test": self.name,
            "description": self.description,
            "chunks": self.chunks
        }

class Prompt(StructuredNode):
    uid = UniqueIdProperty()
    text = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    document =  graph.create_relationship_to(Documents,'HAS_MANY')

class SubProcess(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    document = graph.create_relationship_to(Documents,'HAS_A')
    
class Process(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    subprocess = graph.create_relationship_to(SubProcess,'HAS_A')
    machine = graph.create_relationship_to(Machines,'HAS_A')

class Subject(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    process = graph.create_relationship_to(Process,'HAS_A')

class Topic(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    subject = graph.create_relationship_to(Subject,'HAS_A')

