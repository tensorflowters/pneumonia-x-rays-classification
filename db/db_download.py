from pymongo import MongoClient
from db_access import recreate_tree
import os


db_url = os.getenv('MONGO_URL')

client = MongoClient(db_url)
db = client['zoidberg']
collection = db['chest_xray']

document = collection.find_one()  # You can use filters to find the specific document you want to recreate
recreate_tree(document, 'data', db)
