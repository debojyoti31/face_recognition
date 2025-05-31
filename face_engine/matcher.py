# face_engine/matcher.py

import faiss
import numpy as np
import os
import pickle
import cv2

from face_engine.db import FaceDB
from face_engine.face_model import FaceModel

DATA_DIR = "data"
FAISS_INDEX = os.path.join(DATA_DIR, "face_index.faiss")
ID_MAP = os.path.join(DATA_DIR, "face_ids.pkl")

class FaceMatcher:
    def __init__(self):
        # Use cosine similarity instead of L2 distance
        self.index = faiss.IndexFlatIP(512)  # Inner Product for cosine similarity
        self.id_map = []

        if os.path.exists(FAISS_INDEX) and os.path.exists(ID_MAP):
            self.index = faiss.read_index(FAISS_INDEX)
            with open(ID_MAP, "rb") as f:
                self.id_map = pickle.load(f)

        if self.index.ntotal == 0:
            self._rebuild_index_from_db()

    def _normalize_embedding(self, embedding):
        """Normalize embedding to unit vector for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _rebuild_index_from_db(self):
        print("🧠 Rebuilding FAISS index from faces.sqlite...")
        db = FaceDB()
        model = FaceModel()
        
        # Clear existing index
        self.index = faiss.IndexFlatIP(512)
        self.id_map = []
        
        for name in db.list_faces():
            img_path = db.get_image_path(name)
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    embedding, _ = model.get_face_embedding(img)
                    if embedding is not None:
                        # Normalize embedding
                        normalized_embedding = self._normalize_embedding(embedding)
                        self.index.add(np.array([normalized_embedding]).astype('float32'))
                        self.id_map.append(name)
                        print(f"✅ Added {name} to index")
        
        print(f"🎯 Index rebuilt with {self.index.ntotal} faces")
        self.save()

    def add_face(self, embedding, name):
        # Normalize embedding before adding
        normalized_embedding = self._normalize_embedding(embedding)
        self.index.add(np.array([normalized_embedding]).astype('float32'))
        self.id_map.append(name)
        self.save()

    def search(self, embedding, k=1):
        if self.index.ntotal == 0:
            print("⚠️ No faces in index!")
            return None, None
        
        # Normalize query embedding
        normalized_embedding = self._normalize_embedding(embedding)
        
        # Search (higher scores are better for cosine similarity)
        D, I = self.index.search(np.array([normalized_embedding]).astype('float32'), k)
        
        # Convert cosine similarity to distance (1 - similarity)
        similarity_score = D[0][0]
        distance_score = 1.0 - similarity_score
        
        print(f"🔍 Search result: {self.id_map[I[0][0]]}, similarity: {similarity_score:.3f}, distance: {distance_score:.3f}")
        
        return self.id_map[I[0][0]], distance_score

    def delete_face(self, name):
        indices = [i for i, n in enumerate(self.id_map) if n == name]
        if indices:
            keep = [i for i in range(len(self.id_map)) if i not in indices]
            new_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            self.index = faiss.IndexFlatIP(512)
            self.id_map = [self.id_map[i] for i in keep]
            
            # Normalize reconstructed embeddings
            normalized_embeddings = []
            for i in keep:
                embedding = new_embeddings[i]
                normalized_embeddings.append(self._normalize_embedding(embedding))
            
            if normalized_embeddings:
                self.index.add(np.array(normalized_embeddings).astype('float32'))
            self.save()

    def save(self):
        faiss.write_index(self.index, FAISS_INDEX)
        with open(ID_MAP, "wb") as f:
            pickle.dump(self.id_map, f)
