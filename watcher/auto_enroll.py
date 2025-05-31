# watcher/auto_enroll.py

import os
import pickle
import cv2

from face_engine.face_model import FaceModel
from face_engine.matcher import FaceMatcher
from face_engine.db import FaceDB

ENROLL_DIR = "enroll_folder"
CACHE_FILE = os.path.join("data", "enrolled_paths.pkl")
os.makedirs("data", exist_ok=True)

class AutoEnroller:
    def __init__(self):
        self.model = FaceModel()
        self.matcher = FaceMatcher()
        self.db = FaceDB()
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return set(pickle.load(f))
        return set()

    def _save_cache(self):
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(list(self.cache), f)

    def _get_current_folders(self):
        """Get list of current person folders in enroll directory"""
        if not os.path.exists(ENROLL_DIR):
            return set()
        
        current_folders = set()
        for item in os.listdir(ENROLL_DIR):
            item_path = os.path.join(ENROLL_DIR, item)
            if os.path.isdir(item_path):
                current_folders.add(item)
        return current_folders



    def _enroll_new_faces(self):
        """Enroll new faces from existing folders - save only first image per person for UI"""
        updated = False
        
        if not os.path.exists(ENROLL_DIR):
            print(f"📁 Enroll directory doesn't exist: {ENROLL_DIR}")
            return False
            
        for person_name in os.listdir(ENROLL_DIR):
            person_folder = os.path.join(ENROLL_DIR, person_name)
            if not os.path.isdir(person_folder):
                continue

            person_already_enrolled = person_name in self.db.list_faces()
            first_image_saved = False
            images_processed = 0

            for fname in os.listdir(person_folder):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                    
                fpath = os.path.join(person_folder, fname)
                if fpath in self.cache:
                    continue

                img = cv2.imread(fpath)
                if img is None:
                    print(f"⚠️ Could not read image: {fpath}")
                    continue

                embedding, bbox = self.model.get_face_embedding(img)
                if embedding is not None:
                    # Always add embedding to FAISS index
                    self.matcher.add_face(embedding, person_name)
                    
                    # Save image copy only for the FIRST successful face detection per person
                    if not person_already_enrolled and not first_image_saved:
                        saved_path = os.path.join("data", f"{person_name}_display.jpg")
                        cv2.imwrite(saved_path, img)
                        self.db.add_face(person_name, saved_path)
                        first_image_saved = True
                        print(f"✅ Enrolled {person_name} - saved display image from {fname}")
                    elif not first_image_saved:
                        # Person exists but no display image yet
                        saved_path = os.path.join("data", f"{person_name}_display.jpg")
                        cv2.imwrite(saved_path, img)
                        # Update database with display image path
                        self.db.add_face(person_name, saved_path)
                        first_image_saved = True
                        print(f"✅ Added display image for existing person {person_name} from {fname}")
                    else:
                        print(f"✅ Added embedding for {person_name} from {fname} (no image saved)")
                    
                    self.cache.add(fpath)
                    images_processed += 1
                    updated = True
                else:
                    print(f"⚠️ No face detected in: {fpath}")

            if images_processed > 0:
                print(f"🎯 Total embeddings added for {person_name}: {images_processed}")

        return updated

    def _remove_deleted_faces(self):
        """Remove faces from database and index if their folders no longer exist"""
        current_folders = self._get_current_folders()
        enrolled_names = set(self.db.list_faces())
        
        # Find names that are in database but no longer have folders
        deleted_names = enrolled_names - current_folders
        
        if deleted_names:
            print(f"🗑️ Removing deleted persons: {list(deleted_names)}")
            
            for name in deleted_names:
                # Remove the single display image for this person
                display_image_path = os.path.join("data", f"{name}_display.jpg")
                if os.path.exists(display_image_path):
                    try:
                        os.remove(display_image_path)
                        print(f"  📁 Deleted display image: {display_image_path}")
                    except OSError as e:
                        print(f"  ⚠️ Could not delete {display_image_path}: {e}")
                else:
                    print(f"  ℹ️ No display image found for {name}")
                
                # Remove from database
                self.db.delete_face(name)
                print(f"  🗄️ Removed {name} from database")
                
                # Remove ALL embeddings for this person from FAISS index
                self.matcher.delete_face(name)
                print(f"  🔍 Removed {name} from search index")
            
            # Clean up cache - remove paths for deleted persons
            updated_cache = set()
            for cached_path in self.cache:
                try:
                    person_name = cached_path.split(os.sep)[-2]  # Get folder name
                    if person_name in current_folders:
                        updated_cache.add(cached_path)
                except (IndexError, AttributeError):
                    updated_cache.add(cached_path)
            
            self.cache = updated_cache
            print(f"  🧹 Cleaned cache: {len(self.cache)} entries remaining")
            
            return True
        return False



    def enroll_once(self):
        """Main enrollment function - handles both adding and removing faces"""
        print("🔄 Starting auto-enrollment check...")
        
        # First, remove faces for deleted folders
        removed = self._remove_deleted_faces()
        
        # Then, enroll new faces
        added = self._enroll_new_faces()
        
        # Save cache if anything changed
        if removed or added:
            self._save_cache()
            print("💾 Cache updated")
        
        if not removed and not added:
            print("✨ No changes detected - all up to date")
        
        print(f"🎯 Total enrolled faces: {self.matcher.index.ntotal}")
        print("─" * 50)
