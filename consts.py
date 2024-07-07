from src.api.dao.graph import SubProcess, Subject, Machines, Process, Topic, DocumentChunks, Documents, Prompt, Generic

node_mapping = {
    'topic': Topic,
    'subject': Subject,
    'process': Process,
    'subprocess': SubProcess,
    'prompt': Prompt,
    'documents': Documents,
    'machines': Machines,
    'document_chunks': DocumentChunks,
    'generic': Generic
}

# node_relationship_mapping = {
#     'topic': [Subject],
#     'subject': [Process],
#     'process': [SubProcess, Machines],
#     'subprocess': [Documents],
#     'prompt': [Documents],
#     'documents': [DocumentChunks]
# }