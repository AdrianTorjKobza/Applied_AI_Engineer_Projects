# Convert txt & images to Vector and Store them.

from qdrant_client import QdrantClient # Database engine allowing to search based on meaning rather than just keywords.
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, collection_name="tech_manuals"):
        self.client = QdrantClient(path="storage/qdrant") # Save the database locally.
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # Small AI model that coverts text into vector.
        self.collection = collection_name
        
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config = VectorParams(size=384, distance=Distance.COSINE),
            )

    # Update or Insert.
    def upsert_data(self, text_chunks, image_chunks):
        points = []
        # Index Text.
        for item in text_chunks:
            vector = self.model.encode(item['content']).tolist()
            points.append(PointStruct(id=hash(item['id']) % 10**8, vector=vector, 
                          payload={"content": item['content'], "type": "text"})) # We store the original tet so we can show it to the user later.
        
        # Index Image Captions.
        for item in image_chunks:
            vector = self.model.encode(item['caption']).tolist()
            points.append(PointStruct(id=hash(item['id']) % 10**8, vector=vector, 
                          payload={"content": item['caption'], "path": item['path'], "type": "image"}))
            
        self.client.upsert(self.collection, points)

    # Search in the DB.
    def search(self, query, limit=3):
        # Generate embedding for the user's query.
        vector = self.model.encode(query).tolist()

        # Qdrant looks through all stored vectors and finds the top 3 (limit=3) that have the closest mathematical "distance" to the query vector.
        search_result = self.client.query_points(
            collection_name = self.collection,
            query = vector,
            limit = limit
        )

        return search_result.points