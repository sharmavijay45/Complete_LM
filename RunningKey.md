python main.py --pdf-dir "C:\Users\abc1\Documents\Datasets\23 july"
python main.py --pdf-dir "C:\Users\abc1\Documents\Datasets\26 july"


docker run -d -p 6333:6333 -v /c/qdrant_data:/qdrant/storage qdrant/qdrant



for backup (powershell)
docker run -d -p 6333:6333 ^
  -v C:\qdrant_data_backup:/qdrant/storage ^
  --name qdrant_restored qdrant/qdrant
