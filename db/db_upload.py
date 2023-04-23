from pymongo import MongoClient
from db_access import create_tree
import os


db_url = os.getenv('MONGO_URL')

client = MongoClient(db_url)
db = client['zoidberg']
collection = db['chest_xray']

directory_tree = create_tree('chest_xray', db)
collection.insert_one(directory_tree)
